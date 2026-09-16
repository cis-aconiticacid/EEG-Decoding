"""Mathematical training helpers for the single shared D081 model."""

from __future__ import annotations

import math
from typing import Any, Callable

import torch
from torch.nn import functional as F

from d081_joint_waveform_frequency import latent_loss


def warmup_cosine(step: int, steps_per_round: int, total_rounds: int,
                  warmup_rounds: int, peak: float, minimum: float) -> float:
    total = steps_per_round * total_rounds
    warm = steps_per_round * warmup_rounds
    if not 1 <= step <= total:
        raise ValueError(f"scheduler step {step} outside 1..{total}")
    if step <= warm:
        return peak * step / warm
    progress = (step - warm) / (total - warm)
    return minimum + 0.5 * (peak - minimum) * (1.0 + math.cos(math.pi * progress))


def mask_ratio(round_index: int) -> float:
    if not 1 <= round_index <= 40:
        raise ValueError("stage A round outside 1..40")
    if round_index <= 30:
        return 0.15 + (0.50 - 0.15) * (round_index - 1) / 29
    return 0.50


def hidden_patch_mse(prediction: torch.Tensor, clean_standardized_waveform: torch.Tensor,
                     temporal_mask: torch.Tensor) -> torch.Tensor:
    if prediction.ndim != 4 or prediction.shape[-2:] != (8, 50):
        raise ValueError("reconstruction must be [B,C,8,50]")
    target = clean_standardized_waveform.detach().reshape_as(prediction).float()
    selected = temporal_mask[:, None, :, None].expand_as(prediction)
    if not selected.any():
        raise ValueError("masked reconstruction has no hidden samples")
    return (prediction.float() - target).square()[selected].mean()


def balanced_gate_bce(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
    labels = labels.float()
    positive = labels == 1
    negative = labels == 0
    if not positive.any() or not negative.any():
        raise ValueError("Gate update requires both visual and rest examples")
    positive_loss = F.binary_cross_entropy_with_logits(logits[positive].float(), labels[positive], reduction="mean")
    negative_loss = F.binary_cross_entropy_with_logits(logits[negative].float(), labels[negative], reduction="mean")
    loss = 0.5 * positive_loss + 0.5 * negative_loss
    return loss, {"gate_loss": loss.detach(), "positive_bce": positive_loss.detach(),
                  "negative_bce": negative_loss.detach(), "positive_count": int(positive.sum()),
                  "negative_count": int(negative.sum())}


def gradient_cache_latent_step(
    forwards: list[Callable[[], torch.Tensor]], targets: torch.Tensor, image_ids: torch.Tensor,
    temperature: float = 0.07, contrastive_weight: float = 0.1,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Exact two-pass gradient cache over all logical-batch predictions.

    Each callable must recreate one physical microbatch.  CPU and CUDA RNG
    states are restored so dropout masks in the differentiable pass exactly
    match those used to construct the global 70-sample contrastive loss.
    """
    predictions = []
    rng_states = []
    for forward in forwards:
        cpu_state = torch.get_rng_state()
        cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        rng_states.append((cpu_state, cuda_state))
        with torch.no_grad():
            predictions.append(forward().float())
    logical = torch.cat(predictions).detach().requires_grad_(True)
    if len(logical) != len(targets) or len(logical) != len(image_ids):
        raise ValueError("gradient-cache logical batch arrays differ")
    loss, parts = latent_loss(logical, targets.float(), image_ids, temperature, contrastive_weight)
    loss.backward()
    logical_gradient = logical.grad.detach()
    sizes = [len(value) for value in predictions]
    del predictions, logical
    offset = 0
    for forward, (cpu_state, cuda_state), size in zip(forwards, rng_states, sizes):
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
        prediction = forward()
        torch.autograd.backward(prediction, logical_gradient[offset:offset + size].to(prediction.dtype))
        offset += size
    if offset != len(logical_gradient):
        raise AssertionError("gradient-cache chunk coverage differs")
    return loss.detach(), parts


def gradient_cache_self_test() -> dict[str, Any]:
    torch.manual_seed(17)
    module_direct = torch.nn.Sequential(torch.nn.Linear(7, 11), torch.nn.GELU(), torch.nn.Dropout(0.2),
                                        torch.nn.Linear(11, 12))
    module_cache = torch.nn.Sequential(torch.nn.Linear(7, 11), torch.nn.GELU(), torch.nn.Dropout(0.2),
                                       torch.nn.Linear(11, 12))
    module_cache.load_state_dict(module_direct.state_dict())
    x = torch.randn(6, 7)
    target = torch.randn(6, 3, 4)
    ids = torch.arange(6)
    initial_rng = torch.get_rng_state()
    torch.set_rng_state(initial_rng)
    direct = torch.cat([module_direct(x[lo:hi]).reshape(hi - lo, 3, 4)
                        for lo, hi in ((0, 2), (2, 4), (4, 6))])
    direct_loss, _ = latent_loss(direct, target, ids)
    direct_loss.backward()
    torch.set_rng_state(initial_rng)
    forwards = [lambda lo=lo, hi=hi: module_cache(x[lo:hi]).reshape(hi - lo, 3, 4)
                for lo, hi in ((0, 2), (2, 4), (4, 6))]
    cache_loss, _ = gradient_cache_latent_step(forwards, target, ids)
    maximum = max(float((left.grad - right.grad).abs().max().detach())
                  for left, right in zip(module_direct.parameters(), module_cache.parameters()))
    direct_value = float(direct_loss.detach())
    cache_value = float(cache_loss.detach())
    if abs(direct_value - cache_value) > 1e-7 or maximum > 1e-6:
        raise AssertionError(f"gradient cache differs: loss={direct_loss}/{cache_loss} grad={maximum}")
    return {"status": "PASS", "loss_absolute_difference": abs(direct_value - cache_value),
            "maximum_parameter_gradient_difference": maximum, "dropout_rng_replayed": True}


__all__ = ["warmup_cosine", "mask_ratio", "hidden_patch_mse", "balanced_gate_bce",
           "gradient_cache_latent_step", "gradient_cache_self_test"]
