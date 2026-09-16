"""D089 masked-waveform reconstruction and image-latent retrieval models.

There is deliberately no JEPA/EMA target encoder and no category classifier in
this module.  The pretext target is the masked waveform; the downstream target
is a frozen per-image spatial latent.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


DIM = 384
HEADS = 8
DEPTH = 8
PATCHES = 8
PATCH_SAMPLES = 50
SAMPLES = PATCHES * PATCH_SAMPLES
CHANNELS = 62
FREQ_BINS = 32
FREQ_PATCHES = 8
FREQUENCY_FUSION_SCALE = 0.1
IMAGE_TOKENS = 256
IMAGE_DIM = 1024
ROUTES = ("local_conv", "global_conv", "separate_conv", "shared_conv")
Route = Literal["local_conv", "global_conv", "separate_conv", "shared_conv"]


def periodic_hann_log_psd(waveform: torch.Tensor) -> torch.Tensor:
    """Return 2.5--80 Hz log-power [B,C,32] from [B,C,400]."""

    value = waveform.float() - waveform.float().mean(dim=-1, keepdim=True)
    window = torch.hann_window(SAMPLES, periodic=True, device=value.device)
    transformed = torch.fft.rfft(value * window, dim=-1)
    density = 2.0 * transformed.abs().square() / (1000.0 * window.square().sum())
    return density[..., 1:33].clamp_min(1e-30).log10()


def masked_waveform(waveform: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
    patches = waveform.reshape(len(waveform), CHANNELS, PATCHES, PATCH_SAMPLES)
    visible = torch.where(hidden[:, None, :, None], torch.zeros_like(patches), patches)
    return visible.reshape_as(waveform)


class ResidualQKVDepthwiseConv(nn.Module):
    def __init__(self, dim: int = DIM):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim) for _ in range(3)]
        )

    def forward(self, qkv: tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        return tuple(
            value + layer(value.transpose(1, 2)).transpose(1, 2)
            for value, layer in zip(qkv, self.layers)
        )


def apply_rope(value: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    half = value.shape[-1] // 2
    if value.shape[-1] % 2:
        raise ValueError("RoPE head width must be even")
    inverse = 1.0 / (10000 ** (torch.arange(half, device=value.device).float() / half))
    angle = positions.float()[:, None] * inverse[None]
    cos = angle.cos()[None, None]
    sin = angle.sin()[None, None]
    first, second = value[..., :half], value[..., half:]
    return torch.cat((first * cos - second * sin, second * cos + first * sin), dim=-1)


class DualRangeAttention(nn.Module):
    """Parallel radius-1 and full temporal attention with routed post-QKV conv."""

    def __init__(self, route: Route, dim: int = DIM, heads: int = HEADS, dropout: float = 0.1):
        super().__init__()
        if route not in ROUTES:
            raise ValueError(route)
        self.route = route
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.dropout = float(dropout)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.conv_primary = ResidualQKVDepthwiseConv(dim)
        self.conv_secondary = ResidualQKVDepthwiseConv(dim)
        self.gate = nn.Linear(dim, 1)
        self.output = nn.Linear(dim, dim)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        if route != "separate_conv":
            self.conv_secondary.requires_grad_(False)

    def _heads(self, value: torch.Tensor) -> torch.Tensor:
        return value.reshape(len(value), value.shape[1], self.heads, self.head_dim).permute(0, 2, 1, 3)

    def _attend(self, qkv: tuple[torch.Tensor, torch.Tensor, torch.Tensor], local: bool) -> torch.Tensor:
        q, k, v = (self._heads(item) for item in qkv)
        positions = torch.arange(q.shape[-2], device=q.device)
        q, k = apply_rope(q, positions), apply_rope(k, positions)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if local:
            distance = (positions[:, None] - positions[None, :]).abs()
            scores = scores.masked_fill(distance[None, None] > 1, torch.finfo(scores.dtype).min)
        probability = F.softmax(scores.float(), dim=-1).to(scores.dtype)
        probability = F.dropout(probability, self.dropout, self.training)
        result = torch.matmul(probability, v).permute(0, 2, 1, 3)
        return result.reshape(len(q), q.shape[-2], self.dim)

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        base = tuple(self.qkv(value).chunk(3, dim=-1))
        if self.route == "local_conv":
            local_qkv, global_qkv = self.conv_primary(base), base
        elif self.route == "global_conv":
            local_qkv, global_qkv = base, self.conv_primary(base)
        elif self.route == "separate_conv":
            local_qkv, global_qkv = self.conv_primary(base), self.conv_secondary(base)
        else:
            shared = self.conv_primary(base)
            local_qkv = global_qkv = shared
        local = self._attend(local_qkv, local=True)
        global_ = self._attend(global_qkv, local=False)
        alpha = torch.sigmoid(self.gate(value.float())).to(value.dtype)
        return self.output(alpha * local + (1.0 - alpha) * global_), alpha


class DualAxialBlock(nn.Module):
    def __init__(self, route: Route, dim: int = DIM, heads: int = HEADS, dropout: float = 0.1):
        super().__init__()
        self.time_norm = nn.LayerNorm(dim)
        self.time_attention = DualRangeAttention(route, dim, heads, dropout)
        self.time_drop = nn.Dropout(dropout)
        kwargs = dict(d_model=dim, nhead=heads, dim_feedforward=3 * dim, dropout=dropout,
                      activation="gelu", batch_first=True, norm_first=True)
        self.spatial = nn.TransformerEncoderLayer(**kwargs)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, 3 * dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(3 * dim, dim))
        self.ffn_drop = nn.Dropout(dropout)

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, channels, times, dim = value.shape
        temporal = value.reshape(batch * channels, times, dim)
        attended, gate = self.time_attention(self.time_norm(temporal))
        temporal = temporal + self.time_drop(attended)
        temporal = temporal + self.ffn_drop(self.ffn(self.ffn_norm(temporal)))
        value = temporal.reshape(batch, channels, times, dim)
        spatial = value.transpose(1, 2).reshape(batch * times, channels, dim)
        spatial = self.spatial(spatial)
        value = spatial.reshape(batch, times, channels, dim).transpose(1, 2).contiguous()
        return value, gate.reshape(batch, channels, times, 1)


class D089Encoder(nn.Module):
    def __init__(
        self,
        route: Route,
        dropout: float = 0.1,
        activation_checkpointing: bool = False,
        frequency_fusion_scale: float = FREQUENCY_FUSION_SCALE,
    ):
        super().__init__()
        self.route = route
        self.activation_checkpointing = activation_checkpointing
        self.frequency_fusion_scale = float(frequency_fusion_scale)
        self.wave_patch = nn.Sequential(nn.Linear(PATCH_SAMPLES, DIM), nn.GELU(), nn.Linear(DIM, DIM))
        self.mask_token = nn.Parameter(torch.empty(DIM))
        self.channel_identity = nn.Embedding(CHANNELS, DIM)
        self.time_identity = nn.Embedding(PATCHES, DIM)
        self.frequency_identity = nn.Embedding(FREQ_PATCHES, DIM)
        self.frequency_patch = nn.Linear(4, DIM)
        self.frequency_q_norm = nn.LayerNorm(DIM)
        self.frequency_kv_norm = nn.LayerNorm(DIM)
        self.frequency_attention = nn.MultiheadAttention(DIM, HEADS, dropout=dropout, batch_first=True)
        self.frequency_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([DualAxialBlock(route, dropout=dropout) for _ in range(DEPTH)])
        self.final_norm = nn.LayerNorm(DIM)
        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.channel_identity.weight, std=0.02)
        nn.init.trunc_normal_(self.time_identity.weight, std=0.02)
        nn.init.trunc_normal_(self.frequency_identity.weight, std=0.02)

    def frontend(
        self,
        waveform: torch.Tensor,
        hidden: torch.Tensor,
        frequency_mean: torch.Tensor,
        frequency_scale: torch.Tensor,
    ) -> torch.Tensor:
        visible = masked_waveform(waveform, hidden)
        patches = visible.reshape(len(waveform), CHANNELS, PATCHES, PATCH_SAMPLES)
        content = self.wave_patch(patches)
        identities = self.channel_identity.weight[None, :, None] + self.time_identity.weight[None, None]
        clear = content + identities
        masked = self.mask_token[None, None, None] + identities
        temporal = torch.where(hidden[:, None, :, None], masked, clear)

        spectral = (periodic_hann_log_psd(visible) - frequency_mean[None]) / frequency_scale[None]
        frequency = self.frequency_patch(spectral.reshape(len(waveform), CHANNELS, FREQ_PATCHES, 4))
        frequency = frequency + self.channel_identity.weight[None, :, None] + self.frequency_identity.weight[None, None]
        query = self.frequency_q_norm(temporal).reshape(len(waveform) * CHANNELS, PATCHES, DIM)
        memory = self.frequency_kv_norm(frequency).reshape(len(waveform) * CHANNELS, FREQ_PATCHES, DIM)
        fused = self.frequency_attention(query, memory, memory, need_weights=False)[0]
        frequency_delta = self.frequency_drop(fused.reshape(len(waveform), CHANNELS, PATCHES, DIM))
        return temporal + self.frequency_fusion_scale * frequency_delta

    def forward(
        self,
        waveform: torch.Tensor,
        hidden: torch.Tensor,
        frequency_mean: torch.Tensor,
        frequency_scale: torch.Tensor,
        return_gates: bool = False,
    ):
        value = self.frontend(waveform, hidden, frequency_mean, frequency_scale)
        gates = []
        for block in self.blocks:
            if self.activation_checkpointing and self.training and torch.is_grad_enabled() and not return_gates:
                value, gate = checkpoint(block, value, use_reentrant=False, preserve_rng_state=True)
            else:
                value, gate = block(value)
            if return_gates:
                gates.append(gate)
        value = self.final_norm(value)
        return (value, gates) if return_gates else value


class MaskedWaveformReconstructor(nn.Module):
    def __init__(self, encoder: D089Encoder):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(nn.Linear(DIM, DIM), nn.GELU(), nn.Linear(DIM, PATCH_SAMPLES))

    def forward(self, waveform, hidden, frequency_mean, frequency_scale):
        encoded = self.encoder(waveform, hidden, frequency_mean, frequency_scale)
        return self.head(encoded)

    def loss(self, waveform, hidden, frequency_mean, frequency_scale):
        target = waveform.reshape(len(waveform), CHANNELS, PATCHES, PATCH_SAMPLES)
        predicted = self(waveform, hidden, frequency_mean, frequency_scale)
        select = hidden[:, None, :, None].expand_as(target)
        wave = F.smooth_l1_loss(predicted.float()[select], target.float()[select], beta=1.0)
        slope_select = hidden[:, None, :, None].expand(len(waveform), CHANNELS, PATCHES, PATCH_SAMPLES - 1)
        slope = F.smooth_l1_loss(
            predicted.float().diff(dim=-1)[slope_select], target.float().diff(dim=-1)[slope_select], beta=1.0
        )
        total = wave + 0.2 * slope
        return total, {"wave": wave.detach(), "slope": slope.detach()}


def fixed_2d_sincos(size: int, dim: int) -> torch.Tensor:
    if dim % 4:
        raise ValueError("2D sinusoidal width must be divisible by four")
    axis = torch.arange(size, dtype=torch.float32)
    frequency = 1.0 / (10000 ** (torch.arange(dim // 4, dtype=torch.float32) / (dim // 4)))
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    values = []
    for coordinate in (y.flatten(), x.flatten()):
        angle = coordinate[:, None] * frequency[None]
        values.extend((angle.sin(), angle.cos()))
    return torch.cat(values, dim=-1)


class QueryBlock(nn.Module):
    def __init__(self, dim: int = DIM, heads: int = HEADS, dropout: float = 0.1):
        super().__init__()
        self.self_norm = nn.LayerNorm(dim)
        self.self_attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.cross_query_norm = nn.LayerNorm(dim)
        self.cross_memory_norm = nn.LayerNorm(dim)
        self.cross_attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, 3 * dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(3 * dim, dim))

    def forward(self, query: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        normalized = self.self_norm(query)
        query = query + self.self_attention(normalized, normalized, normalized, need_weights=False)[0]
        query = query + self.cross_attention(
            self.cross_query_norm(query), self.cross_memory_norm(memory), self.cross_memory_norm(memory),
            need_weights=False,
        )[0]
        return query + self.ffn(self.ffn_norm(query))


class ImageLatentPredictor(nn.Module):
    def __init__(self, encoder: D089Encoder, query_depth: int = 4, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        self.query = nn.Parameter(torch.empty(IMAGE_TOKENS, DIM))
        self.register_buffer("query_position", fixed_2d_sincos(16, DIM), persistent=True)
        self.blocks = nn.ModuleList([QueryBlock(dropout=dropout) for _ in range(query_depth)])
        self.norm = nn.LayerNorm(DIM)
        self.output = nn.Sequential(nn.Linear(DIM, 512), nn.GELU(), nn.Linear(512, IMAGE_DIM))
        nn.init.normal_(self.query, std=0.02)

    def forward(self, waveform, frequency_mean, frequency_scale):
        hidden = torch.zeros((len(waveform), PATCHES), dtype=torch.bool, device=waveform.device)
        memory = self.encoder(waveform, hidden, frequency_mean, frequency_scale)
        memory = memory.reshape(len(waveform), CHANNELS * PATCHES, DIM)
        query = (self.query + self.query_position)[None].expand(len(waveform), -1, -1)
        for block in self.blocks:
            query = block(query, memory)
        return self.output(self.norm(query))


def image_latent_loss(predicted: torch.Tensor, target: torch.Tensor, temperature: float = 0.07):
    mse = F.mse_loss(predicted.float(), target.float())
    cosine = 1.0 - F.cosine_similarity(predicted.float(), target.float(), dim=-1).mean()
    predicted_flat = normalized_flat(predicted)
    target_flat = normalized_flat(target)
    logits = predicted_flat @ target_flat.T / temperature
    labels = torch.arange(len(predicted), device=predicted.device)
    contrastive = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
    total = mse + 0.1 * cosine + 0.1 * contrastive
    return total, {"mse": mse.detach(), "cosine_loss": cosine.detach(), "contrastive": contrastive.detach()}


def normalized_flat(value: torch.Tensor) -> torch.Tensor:
    """Flatten per-position unit vectors so dot product is mean token cosine."""
    return F.normalize(value.float(), dim=-1, eps=1e-8).flatten(1) / math.sqrt(value.shape[1])


def parameter_counts(model: nn.Module) -> dict[str, int]:
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    }


__all__ = [
    "CHANNELS", "DIM", "FREQ_PATCHES", "IMAGE_DIM", "IMAGE_TOKENS", "PATCHES", "PATCH_SAMPLES",
    "ROUTES", "D089Encoder", "ImageLatentPredictor", "MaskedWaveformReconstructor", "image_latent_loss",
    "masked_waveform", "normalized_flat", "parameter_counts", "periodic_hann_log_psd",
]
