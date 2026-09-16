"""D081 large waveform/frequency encoder and independent resting-state gate.

This module contains no historical-checkpoint loading path.  The main model
accepts train-statistics-standardized 400 ms waveforms and keeps every real
channel and all eight 50 ms temporal patches through the axial trunk.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

DIM = 384
HEADS = 8
AXIAL_DEPTH = 8
QUERY_DEPTH = 4
PATCHES = 8
PATCH_SAMPLES = 50
SAMPLES = PATCHES * PATCH_SAMPLES
FREQUENCY_BINS = 32
FREQUENCY_PATCHES = 8
DATASET_CHANNELS = {"imagenet": 62, "things": 63}


def grid_encoding(size: int = 16, dim: int = DIM) -> torch.Tensor:
    if dim % 4:
        raise ValueError("2D encoding width must be divisible by four")
    axis = torch.arange(size, dtype=torch.float32)
    omega = 1.0 / (10000 ** (torch.arange(dim // 4).float() / (dim // 4)))
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    return torch.cat([function(value.reshape(-1, 1) * omega)
                      for value in (x, y) for function in (torch.sin, torch.cos)], dim=-1)


class AxialBlock(nn.Module):
    def __init__(self, dim: int = DIM, heads: int = HEADS, dropout: float = 0.1):
        super().__init__()
        kwargs = dict(d_model=dim, nhead=heads, dim_feedforward=dim * 3,
                      dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.temporal = nn.TransformerEncoderLayer(**kwargs)
        self.spatial = nn.TransformerEncoderLayer(**kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, times, dim = x.shape
        x = self.temporal(x.reshape(batch * channels, times, dim)).reshape(batch, channels, times, dim)
        x = self.spatial(x.transpose(1, 2).reshape(batch * times, channels, dim))
        return x.reshape(batch, times, channels, dim).transpose(1, 2).contiguous()


class QueryBlock(nn.Module):
    def __init__(self, dim: int = DIM, heads: int = HEADS, dropout: float = 0.1):
        super().__init__()
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(3)])
        self.self_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 3), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(dim * 3, dim))
        self.drop = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        value = self.norms[0](query)
        query = query + self.drop(self.self_attn(value, value, value, need_weights=False)[0])
        query = query + self.drop(self.cross_attn(self.norms[1](query), memory, memory, need_weights=False)[0])
        return query + self.drop(self.ffn(self.norms[2](query)))


def normalized_flat(value: torch.Tensor) -> torch.Tensor:
    return F.normalize(value.float(), dim=-1, eps=1e-8).flatten(1) / math.sqrt(value.shape[1])


def latent_loss(predicted: torch.Tensor, targets: torch.Tensor, image_ids: torch.Tensor,
                temperature: float = 0.07, contrastive_weight: float = 0.1
                ) -> tuple[torch.Tensor, dict[str, Any]]:
    if predicted.shape != targets.shape or predicted.ndim != 3:
        raise ValueError("prediction and target must share [B,P,D] shape")
    predicted, targets = predicted.float(), targets.detach().float()
    mse = (predicted - targets).square().mean()
    unique, inverse = torch.unique(image_ids, sorted=True, return_inverse=True)
    if len(unique) < 2:
        contrastive = predicted.sum() * 0.0
    else:
        first = torch.stack([torch.nonzero(inverse == index, as_tuple=False)[0, 0]
                             for index in range(len(unique))])
        logits = normalized_flat(predicted) @ normalized_flat(targets[first]).T / temperature
        forward = F.cross_entropy(logits, inverse)
        positive = inverse[:, None].eq(torch.arange(len(unique), device=inverse.device)[None])
        reverse = (torch.logsumexp(logits, dim=0)
                   - torch.logsumexp(logits.masked_fill(~positive, -torch.inf), dim=0)).mean()
        contrastive = (forward + reverse) * 0.5
    loss = mse + contrastive_weight * contrastive
    return loss, {"latent_mse": mse.detach(), "contrastive": contrastive.detach(),
                  "loss": loss.detach(), "unique_images": len(unique)}


def periodic_hann_psd_2p5_80(standardized_waveform: torch.Tensor) -> torch.Tensor:
    """Return contract PSD log10 values [B,C,32] from [B,C,400]."""
    if standardized_waveform.ndim != 3 or standardized_waveform.shape[-1] != SAMPLES:
        raise ValueError(f"expected [B,C,{SAMPLES}], got {tuple(standardized_waveform.shape)}")
    waveform = standardized_waveform.float()
    waveform = waveform - waveform.mean(dim=-1, keepdim=True)
    window = torch.hann_window(SAMPLES, periodic=True, device=waveform.device, dtype=waveform.dtype)
    transformed = torch.fft.rfft(waveform * window, dim=-1)
    density = 2.0 * transformed.abs().square() / (1000.0 * window.square().sum())
    return density[..., 1:33].clamp_min(1e-30).log10()


def apply_synchronous_temporal_mask(
    standardized_waveform: torch.Tensor, temporal_mask: torch.Tensor
) -> torch.Tensor:
    """Set the same selected 50 ms positions to zero in every channel."""
    if temporal_mask.dtype != torch.bool or temporal_mask.shape != (len(standardized_waveform), PATCHES):
        raise ValueError(f"mask must be bool [B,{PATCHES}]")
    patches = standardized_waveform.reshape(len(standardized_waveform), standardized_waveform.shape[1], PATCHES, PATCH_SAMPLES)
    return torch.where(temporal_mask[:, None, :, None], torch.zeros((), device=patches.device, dtype=patches.dtype), patches).reshape_as(standardized_waveform)


class D081JointEncoder(nn.Module):
    """Shared large encoder with dataset-specific channel identity tables."""

    def __init__(self, dropout: float = 0.1, activation_checkpointing: bool = False):
        super().__init__()
        self.activation_checkpointing = bool(activation_checkpointing)
        self.wave_patch = nn.Sequential(nn.Linear(PATCH_SAMPLES, DIM), nn.GELU(), nn.Linear(DIM, DIM))
        self.channel_identity = nn.ModuleDict({
            dataset: nn.Embedding(channels, DIM) for dataset, channels in DATASET_CHANNELS.items()
        })
        self.time_index = nn.Embedding(PATCHES, DIM)
        self.frequency_index = nn.Embedding(FREQUENCY_PATCHES, DIM)
        self.frequency_patch = nn.Linear(4, DIM)
        self.frequency_q_norm = nn.LayerNorm(DIM)
        self.frequency_kv_norm = nn.LayerNorm(DIM)
        self.frequency_cross_attention = nn.MultiheadAttention(DIM, HEADS, dropout=dropout, batch_first=True)
        self.frequency_drop = nn.Dropout(dropout)
        self.axial_blocks = nn.ModuleList([AxialBlock(DIM, HEADS, dropout) for _ in range(AXIAL_DEPTH)])
        self.final_norm = nn.LayerNorm(DIM)
        self.mask_token = nn.Parameter(torch.empty(DIM))
        self.mask_head = nn.Linear(DIM, PATCH_SAMPLES)
        nn.init.trunc_normal_(self.time_index.weight, std=0.02)
        nn.init.trunc_normal_(self.frequency_index.weight, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)

    @staticmethod
    def _check_dataset(dataset: str, channels: int) -> None:
        if dataset not in DATASET_CHANNELS:
            raise ValueError(f"unknown D081 dataset {dataset!r}")
        if channels != DATASET_CHANNELS[dataset]:
            raise ValueError(f"{dataset} expects {DATASET_CHANNELS[dataset]} channels, got {channels}")

    def frontend_tokens(
        self,
        standardized_waveform: torch.Tensor,
        dataset: str,
        temporal_mask: torch.Tensor | None = None,
        frequency_mean: torch.Tensor | None = None,
        frequency_scale: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return raw-time and time+frequency tokens, each [B,C,8,384].

        This is the production training path.  ``raw_time_tokens`` include the
        dataset channel identity and temporal index. ``fused_tokens`` are the
        residual cross-attention result before the axial trunk.
        """
        if standardized_waveform.ndim != 3 or standardized_waveform.shape[-1] != SAMPLES:
            raise ValueError(f"expected [B,C,{SAMPLES}]")
        batch, channels, _ = standardized_waveform.shape
        self._check_dataset(dataset, channels)
        if temporal_mask is None:
            visible_waveform = standardized_waveform
        else:
            visible_waveform = apply_synchronous_temporal_mask(standardized_waveform, temporal_mask)

        temporal_patches = visible_waveform.reshape(batch, channels, PATCHES, PATCH_SAMPLES)
        temporal = self.wave_patch(temporal_patches)
        if temporal_mask is not None:
            temporal = torch.where(temporal_mask[:, None, :, None], self.mask_token[None, None, None], temporal)
        identity = self.channel_identity[dataset].weight[None, :, None, :]
        temporal = temporal + identity + self.time_index.weight[None, None, :, :]
        raw_time_tokens = temporal

        # This is deliberately derived only from visible_waveform.  A caller has
        # no argument through which a clean full-window spectrum can enter.
        spectral = periodic_hann_psd_2p5_80(visible_waveform)
        if (frequency_mean is None) != (frequency_scale is None):
            raise ValueError("frequency_mean and frequency_scale must be supplied together")
        if frequency_mean is not None:
            expected = (channels, FREQUENCY_BINS)
            if tuple(frequency_mean.shape) != expected or tuple(frequency_scale.shape) != expected:
                raise ValueError(f"frequency statistics must be {expected}")
            if not torch.isfinite(frequency_mean).all() or not torch.isfinite(frequency_scale).all() or (frequency_scale <= 0).any():
                raise ValueError("invalid train-only frequency statistics")
            spectral = (spectral - frequency_mean[None]) / frequency_scale[None]
        frequency = self.frequency_patch(spectral.reshape(batch, channels, FREQUENCY_PATCHES, 4))
        frequency = frequency + identity + self.frequency_index.weight[None, None, :, :]
        q = self.frequency_q_norm(temporal).reshape(batch * channels, PATCHES, DIM)
        kv = self.frequency_kv_norm(frequency).reshape(batch * channels, FREQUENCY_PATCHES, DIM)
        fused = self.frequency_cross_attention(q, kv, kv, need_weights=False)[0]
        fused_tokens = temporal + self.frequency_drop(fused.reshape(batch, channels, PATCHES, DIM))
        return raw_time_tokens, fused_tokens

    def encode(
        self,
        standardized_waveform: torch.Tensor,
        dataset: str,
        temporal_mask: torch.Tensor | None = None,
        frequency_mean: torch.Tensor | None = None,
        frequency_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode [B,C,400]; mask is applied before the frequency computation."""
        _, temporal = self.frontend_tokens(
            standardized_waveform, dataset, temporal_mask, frequency_mean, frequency_scale
        )
        for block in self.axial_blocks:
            if self.activation_checkpointing and self.training and torch.is_grad_enabled():
                temporal = checkpoint(block, temporal, use_reentrant=False, preserve_rng_state=True)
            else:
                temporal = block(temporal)
        return self.final_norm(temporal)

    def reconstruct(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.mask_head(self.encode(*args, **kwargs))


class D081ImageHead(nn.Module):
    def __init__(self, dropout: float = 0.1, activation_checkpointing: bool = False):
        super().__init__()
        self.activation_checkpointing = bool(activation_checkpointing)
        self.queries = nn.Parameter(torch.empty(256, DIM))
        nn.init.trunc_normal_(self.queries, std=0.02)
        self.register_buffer("grid_position", grid_encoding(16, DIM))
        self.query_blocks = nn.ModuleList([QueryBlock(DIM, HEADS, dropout) for _ in range(QUERY_DEPTH)])
        self.projector = nn.Sequential(
            nn.LayerNorm(DIM), nn.Linear(DIM, 1024), nn.GELU(), nn.Dropout(dropout), nn.Linear(1024, 1024)
        )

    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        if memory.ndim != 4 or memory.shape[-1] != DIM:
            raise ValueError("memory must be [B,C,8,384]")
        memory = memory.reshape(len(memory), -1, DIM)
        query = (self.queries + self.grid_position)[None].expand(len(memory), -1, -1)
        for block in self.query_blocks:
            if self.activation_checkpointing and self.training and torch.is_grad_enabled():
                query = checkpoint(block, query, memory, use_reentrant=False, preserve_rng_state=True)
            else:
                query = block(query, memory)
        return self.projector(query)


class D081LargeModel(nn.Module):
    def __init__(self, dropout: float = 0.1, activation_checkpointing: bool = False, seed: int = 17):
        super().__init__()
        torch.manual_seed(seed)
        self.encoder = D081JointEncoder(dropout, activation_checkpointing)
        self.image_head = D081ImageHead(dropout, activation_checkpointing)

    def forward(self, standardized_waveform: torch.Tensor, dataset: str,
                frequency_mean: torch.Tensor | None = None,
                frequency_scale: torch.Tensor | None = None) -> torch.Tensor:
        memory = self.encoder.encode(standardized_waveform, dataset, None, frequency_mean, frequency_scale)
        return self.image_head(memory)

    def stage_a(self) -> None:
        self.requires_grad_(True)
        self.image_head.requires_grad_(False)

    def stage_b(self) -> None:
        self.requires_grad_(True)
        self.encoder.mask_token.requires_grad_(False)
        self.encoder.mask_head.requires_grad_(False)


class D081GateNet(nn.Module):
    """ID-free gate operating on the common 0.5-40 Hz clean view."""

    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.wave = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=25, stride=5), nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=9, stride=2), nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.frequency = nn.Sequential(nn.Linear(16, 32), nn.GELU())
        self.classifier = nn.Sequential(
            nn.LayerNorm(192), nn.Linear(192, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1)
        )

    def forward(self, waveform_0p5_40hz: torch.Tensor,
                frequency_mean: torch.Tensor | None = None,
                frequency_scale: torch.Tensor | None = None) -> torch.Tensor:
        if waveform_0p5_40hz.ndim != 3 or waveform_0p5_40hz.shape[-1] != SAMPLES:
            raise ValueError("gate input must be [B,C,400]")
        batch, channels, _ = waveform_0p5_40hz.shape
        wave_features = self.wave(waveform_0p5_40hz.reshape(batch * channels, 1, SAMPLES)).reshape(batch, channels, 64)
        spectrum = periodic_hann_psd_2p5_80(waveform_0p5_40hz)[..., :16]
        if (frequency_mean is None) != (frequency_scale is None):
            raise ValueError("gate frequency statistics must be supplied together")
        if frequency_mean is not None:
            if tuple(frequency_mean.shape) != (channels, 16) or tuple(frequency_scale.shape) != (channels, 16):
                raise ValueError("gate frequency statistics must be [C,16]")
            spectrum = (spectrum - frequency_mean[None]) / frequency_scale[None]
        features = torch.cat((wave_features, self.frequency(spectrum)), dim=-1)
        pooled = torch.cat((features.mean(dim=1), features.amax(dim=1)), dim=-1)
        return self.classifier(pooled).squeeze(-1)


def parameter_audit(model: D081LargeModel) -> dict[str, Any]:
    total = sum(parameter.numel() for parameter in model.parameters())
    model.stage_a()
    stage_a = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    model.stage_b()
    stage_b = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    expected = {"total": 34_331_570, "stage_a_trainable": 24_503_474, "stage_b_trainable": 34_311_936}
    actual = {"total": total, "stage_a_trainable": stage_a, "stage_b_trainable": stage_b}
    if actual != expected:
        raise AssertionError(f"D081 large parameter contract differs: {actual} != {expected}")
    return {**actual, "expected": expected, "status": "PASS"}


def model_state_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def mask_leakage_self_test() -> dict[str, Any]:
    torch.manual_seed(17081)
    model = D081LargeModel(dropout=0.0, seed=17).eval()
    waveform = torch.randn(1, 62, SAMPLES)
    mask = torch.tensor([[False, True, False, False, True, False, False, False]])
    altered = waveform.clone()
    altered.reshape(1, 62, PATCHES, PATCH_SAMPLES)[:, :, mask[0]] = torch.randn(1, 62, int(mask.sum()), PATCH_SAMPLES) * 1000
    with torch.no_grad():
        visible_a = apply_synchronous_temporal_mask(waveform, mask)
        visible_b = apply_synchronous_temporal_mask(altered, mask)
        encoded_a = model.encoder.encode(waveform, "imagenet", mask)
        encoded_b = model.encoder.encode(altered, "imagenet", mask)
    visible_diff = float((visible_a - visible_b).abs().max())
    prediction_diff = float((encoded_a - encoded_b).abs().max())
    hidden_target_diff = float((waveform - altered).abs().amax())
    if hidden_target_diff <= 0 or visible_diff != 0 or prediction_diff > 1e-6:
        raise AssertionError((hidden_target_diff, visible_diff, prediction_diff))
    return {
        "status": "PASS",
        "hidden_target_max_difference": hidden_target_diff,
        "visible_model_input_max_difference": visible_diff,
        "encoded_prediction_max_difference": prediction_diff,
        "scope": "same visible samples and same mask; changed hidden clean targets cannot affect frequency or encoder input",
    }


__all__ = [
    "D081LargeModel", "D081JointEncoder", "D081ImageHead", "D081GateNet",
    "periodic_hann_psd_2p5_80", "apply_synchronous_temporal_mask",
    "parameter_audit", "model_state_sha256", "mask_leakage_self_test", "latent_loss",
]
