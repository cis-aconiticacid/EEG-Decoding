"""D078 frequency-smoothed global noise and frozen sensitive-cell masking."""

from __future__ import annotations

import math
from typing import Any

import torch


ARMS = ("A0", "A1", "A2", "A3")
GLOBAL_SIGMA = {"A0": 0.0, "A1": 0.05, "A2": 0.05, "A3": 0.05}
SENSITIVE_TOTAL_SIGMA = {"A0": 0.0, "A1": 0.05, "A2": 0.15, "A3": 0.15}
MASK_COUNT = {"A0": 0, "A1": 0, "A2": 0, "A3": 124}


def smoothed_standard_noise(
    shape: tuple[int, int, int],
    *,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator,
) -> torch.Tensor:
    """Return unit-variance Gaussian noise smoothed only along 32 frequency bins."""
    if len(shape) != 3 or shape[1:] != (62, 32):
        raise ValueError(f"expected [B,62,32], got {shape}")
    raw = torch.randn(shape, device=device, dtype=dtype, generator=generator)
    out = 0.5 * raw
    out[..., 1:] += 0.25 * raw[..., :-1]
    out[..., :-1] += 0.25 * raw[..., 1:]
    norm = torch.full((32,), math.sqrt(0.25**2 + 0.5**2 + 0.25**2), device=device, dtype=dtype)
    norm[0] = norm[-1] = math.sqrt(0.5**2 + 0.25**2)
    return out / norm


def apply_augmentation(
    x_tokens: torch.Tensor,
    arm: str,
    sensitive_flat_indices: torch.Tensor,
    generator: torch.Generator | None,
    *,
    return_audit: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
    """Apply D078 augmentation to D064-normalized data shaped [B,496,4]."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    if x_tokens.ndim != 3 or x_tokens.shape[1:] != (496, 4):
        raise ValueError(f"expected [B,496,4], got {tuple(x_tokens.shape)}")
    sensitive = sensitive_flat_indices.to(device=x_tokens.device, dtype=torch.long)
    if sensitive.ndim != 1 or len(sensitive) != 248 or len(torch.unique(sensitive)) != 248:
        raise ValueError("sensitive set must contain 248 unique flattened 62x32 indices")
    if int(sensitive.min()) < 0 or int(sensitive.max()) >= 1984:
        raise ValueError("sensitive index outside 62x32 grid")
    if arm == "A0":
        result = x_tokens
        audit = {"arm": arm, "global_noise": False, "extra_sensitive_noise": False, "masked_per_sample": 0}
        return (result, audit) if return_audit else result
    if generator is None:
        raise ValueError("augmented arms require an independent generator")

    grid = x_tokens.reshape(len(x_tokens), 62, 32)
    base = smoothed_standard_noise(tuple(grid.shape), device=grid.device, dtype=grid.dtype, generator=generator)
    output = grid + GLOBAL_SIGMA[arm] * base
    extra_used = arm in {"A2", "A3"}
    if extra_used:
        extra_sigma = math.sqrt(SENSITIVE_TOTAL_SIGMA[arm] ** 2 - GLOBAL_SIGMA[arm] ** 2)
        extra = smoothed_standard_noise(tuple(grid.shape), device=grid.device, dtype=grid.dtype, generator=generator)
        extra_flat = extra.reshape(len(grid), -1)
        output_flat = output.reshape(len(grid), -1)
        output_flat[:, sensitive] += extra_sigma * extra_flat[:, sensitive]
    else:
        output_flat = output.reshape(len(grid), -1)

    masked = MASK_COUNT[arm]
    if masked:
        ranking = torch.rand((len(grid), len(sensitive)), device=grid.device, generator=generator).argsort(dim=1)
        selected = sensitive[ranking[:, :masked]]
        output_flat.scatter_(1, selected, 0.0)
        if not torch.all((output_flat == 0).gather(1, selected)):
            raise AssertionError("sensitive masking failed")
    result = output_flat.reshape_as(x_tokens)
    audit = {
        "arm": arm,
        "global_noise": True,
        "global_sigma": GLOBAL_SIGMA[arm],
        "extra_sensitive_noise": extra_used,
        "sensitive_total_sigma": SENSITIVE_TOTAL_SIGMA[arm],
        "masked_per_sample": masked,
    }
    return (result, audit) if return_audit else result


def self_test() -> dict[str, Any]:
    device = torch.device("cpu")
    sensitive = torch.arange(248, dtype=torch.long)
    zero = torch.zeros(8192, 496, 4)
    results: dict[str, Any] = {}
    for arm in ARMS:
        generator = None if arm == "A0" else torch.Generator().manual_seed(78000)
        out, audit = apply_augmentation(zero, arm, sensitive, generator, return_audit=True)
        flat = out.reshape(len(out), -1)
        if arm == "A0":
            if not torch.equal(out, zero):
                raise AssertionError("A0 is not exact identity")
        elif arm == "A1":
            if not torch.allclose(flat.std(0, correction=0).mean(), torch.tensor(0.05), atol=8e-4):
                raise AssertionError("A1 global standard deviation differs")
        elif arm == "A2":
            selected_sd = flat[:, sensitive].std(0, correction=0).mean()
            other_sd = flat[:, 248:].std(0, correction=0).mean()
            if not torch.allclose(selected_sd, torch.tensor(0.15), atol=0.002) or not torch.allclose(other_sd, torch.tensor(0.05), atol=8e-4):
                raise AssertionError("A2 selected/global standard deviations differ")
        else:
            if not torch.all((flat[:, sensitive] == 0).sum(1) == 124):
                raise AssertionError("A3 must mask exactly 124 sensitive cells per sample")
        results[arm] = audit

    generator1 = torch.Generator().manual_seed(123)
    generator2 = torch.Generator().manual_seed(123)
    first = apply_augmentation(zero[:4], "A3", sensitive, generator1)
    second = apply_augmentation(zero[:4], "A3", sensitive, generator2)
    if not torch.equal(first, second):
        raise AssertionError("augmentation is not reproducible")
    return {"status": "PASS", "shape": [8192, 496, 4], "arms": results, "reproducible": True}
