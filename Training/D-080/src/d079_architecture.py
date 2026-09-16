"""D079 coordinate-removal and learned early-channel-fusion encoders."""

from __future__ import annotations

import hashlib
from typing import Any

import torch
from torch import nn


ARMS = ("B0", "B1", "B2")


class NoCoordinateFrequency(nn.Module):
    """D064 encoder with the complete coordinate projection removed."""

    token_count = 496

    def __init__(self, seed: int):
        super().__init__()
        torch.manual_seed(seed)
        self.projection = nn.Linear(4, 128)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 128))
        nn.init.normal_(self.mask_token, std=0.02)
        self.electrode = nn.Embedding(62, 128)
        self.frequency_position = nn.Embedding(8, 128)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(128, 4, 384, 0.1, activation="gelu", batch_first=True)
            for _ in range(2)
        ])
        self.output = nn.Linear(128, 4)

    def tokens(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1:] != (496, 4):
            raise ValueError("D079 B1 input must be [B,496,4]")
        h = self.projection(x)
        if mask is not None:
            h = torch.where(mask[..., None], self.mask_token, h)
        identity_frequency = (
            self.electrode.weight[:, None, :] + self.frequency_position.weight[None, :, :]
        ).reshape(496, 128)
        return h + identity_frequency

    def encode(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.tokens(x, mask)
        for layer in self.layers:
            h = layer(h)
        return h

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.output(self.encode(x, mask))


class EarlyFusionFrequency(nn.Module):
    """Coordinate-free encoder that learns 16 mixtures over all 62 channels."""

    token_count = 128

    def __init__(self, seed: int):
        super().__init__()
        torch.manual_seed(seed)
        self.projection = nn.Linear(4, 128)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 128))
        nn.init.normal_(self.mask_token, std=0.02)
        self.electrode = nn.Embedding(62, 128)
        self.frequency_position = nn.Embedding(8, 128)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(128, 4, 384, 0.1, activation="gelu", batch_first=True)
            for _ in range(2)
        ])
        self.output = nn.Linear(128, 4)
        self.fusion = nn.Parameter(torch.empty(16, 62))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 79000)
            nn.init.orthogonal_(self.fusion)

    def tokens(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1:] != (496, 4):
            raise ValueError("D079 B2 input must be [B,496,4]")
        batch = len(x)
        h = self.projection(x).reshape(batch, 62, 8, 128)
        if mask is not None:
            h = torch.where(mask.reshape(batch, 62, 8, 1), self.mask_token.reshape(1, 1, 1, 128), h)
        h = h + self.electrode.weight[None, :, None, :]
        h = torch.einsum("kc,bcfd->bkfd", self.fusion, h)
        h = h + self.frequency_position.weight[None, None, :, :]
        return h.reshape(batch, 128, 128)

    def encode(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.tokens(x, mask)
        for layer in self.layers:
            h = layer(h)
        return h

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.output(self.encode(x, mask))


def tensor_sha(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def load_compatible_parent(model: nn.Module, parent_state: dict[str, torch.Tensor], arm: str) -> dict[str, Any]:
    if arm == "B0":
        model.load_state_dict(parent_state, strict=True)
        return {"ignored_parent_keys": [], "new_model_keys": []}
    current = model.state_dict()
    dropped = {"encoder.coordinates", "encoder.position.weight", "encoder.position.bias"}
    absent = set(parent_state) - set(current)
    if absent != dropped:
        raise AssertionError(f"unexpected parent keys removed for {arm}: {sorted(absent)}")
    compatible = {key: value for key, value in parent_state.items() if key in current}
    result = model.load_state_dict(compatible, strict=False)
    expected_missing = {"encoder.fusion"} if arm == "B2" else set()
    if set(result.missing_keys) != expected_missing or result.unexpected_keys:
        raise AssertionError(
            f"parent adaptation differs for {arm}: missing={result.missing_keys}, unexpected={result.unexpected_keys}"
        )
    return {"ignored_parent_keys": sorted(dropped), "new_model_keys": sorted(expected_missing)}


def architecture_record(model: nn.Module, arm: str) -> dict[str, Any]:
    state_names = set(model.state_dict())
    if arm in {"B1", "B2"} and any("position" in name and "frequency_position" not in name for name in state_names):
        raise AssertionError("coordinate position state remains in coordinate-free arm")
    if arm in {"B1", "B2"} and any("coordinates" in name for name in state_names):
        raise AssertionError("coordinate buffer remains in coordinate-free arm")
    encoder = model.encoder
    record = {
        "arm": arm,
        "attention_tokens": int(encoder.token_count if hasattr(encoder, "token_count") else 496),
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "coordinate_projection_parameters": 0 if arm in {"B1", "B2"} else 512,
        "fusion_parameters": int(encoder.fusion.numel()) if arm == "B2" else 0,
    }
    if arm == "B2":
        gram = encoder.fusion.detach() @ encoder.fusion.detach().T
        record.update({
            "fusion_shape": list(encoder.fusion.shape),
            "fusion_initial_sha256": tensor_sha(encoder.fusion),
            "fusion_row_orthogonality_max_error": float((gram - torch.eye(16, device=gram.device)).abs().max()),
            "fusion_all_channel_columns_nonzero": bool((encoder.fusion.detach().abs().sum(0) > 0).all()),
        })
    return record
