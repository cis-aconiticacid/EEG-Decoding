"""D088 subject-0 JEPA models for post-QKV convolution routing ablations."""

from __future__ import annotations

import copy
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
ROUTES = ("local_conv", "global_conv", "separate_conv", "shared_conv")
Route = Literal["local_conv", "global_conv", "separate_conv", "shared_conv"]


def periodic_hann_log_psd(waveform: torch.Tensor) -> torch.Tensor:
    """Return 2.5--80 Hz log-power [B,C,32] from standardized [B,C,400]."""
    waveform = waveform.float() - waveform.float().mean(dim=-1, keepdim=True)
    window = torch.hann_window(SAMPLES, periodic=True, device=waveform.device)
    transformed = torch.fft.rfft(waveform * window, dim=-1)
    density = 2.0 * transformed.abs().square() / (1000.0 * window.square().sum())
    return density[..., 1:33].clamp_min(1e-30).log10()


def mask_waveform(waveform: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
    patches = waveform.reshape(len(waveform), CHANNELS, PATCHES, PATCH_SAMPLES)
    return torch.where(hidden[:, None, :, None], torch.zeros_like(patches), patches).reshape_as(waveform)


class ResidualQKVDepthwiseConv(nn.Module):
    """Separate residual depthwise temporal convolutions for Q, K, and V."""

    def __init__(self, dim: int = DIM):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=True) for _ in range(3)]
        )

    def forward(self, qkv: tuple[torch.Tensor, torch.Tensor, torch.Tensor], valid: torch.Tensor):
        keep = valid[:, :, None].to(qkv[0].dtype)
        output = []
        for value, layer in zip(qkv, self.layers):
            value = value * keep
            convolved = layer(value.transpose(1, 2)).transpose(1, 2)
            output.append((value + convolved) * keep)
        return tuple(output)


def _apply_rope(value: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to [N,H,T,Dh] with integer positions [T]."""
    half = value.shape[-1] // 2
    if value.shape[-1] % 2:
        raise ValueError("head width must be even for RoPE")
    inverse = 1.0 / (10000 ** (torch.arange(half, device=value.device).float() / half))
    angles = positions.to(value.device).float()[:, None] * inverse[None, :]
    cos = angles.cos()[None, None]
    sin = angles.sin()[None, None]
    first, second = value[..., :half], value[..., half:]
    return torch.cat((first * cos - second * sin, second * cos + first * sin), dim=-1)


class DualRangeAttention(nn.Module):
    """Window and full attention with one of four post-QKV convolution routes."""

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

    def _attend(
        self,
        qkv: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        valid: torch.Tensor,
        local: bool,
    ) -> torch.Tensor:
        q, k, v = (self._heads(value) for value in qkv)
        positions = torch.arange(q.shape[-2], device=q.device)
        q, k = _apply_rope(q, positions), _apply_rope(k, positions)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        allowed = valid[:, None, None, :]
        if local:
            distance = (positions[:, None] - positions[None, :]).abs()
            allowed = allowed & (distance[None, None] <= 1)
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        probability = F.softmax(scores.float(), dim=-1).to(scores.dtype)
        probability = F.dropout(probability, self.dropout, self.training)
        result = torch.matmul(probability, v).permute(0, 2, 1, 3).reshape(len(q), q.shape[-2], self.dim)
        return result * valid[:, :, None].to(result.dtype)

    def forward(self, value: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if value.ndim != 3 or valid.shape != value.shape[:2]:
            raise ValueError("attention expects [N,T,D] and valid [N,T]")
        base = tuple(self.qkv(value).chunk(3, dim=-1))
        base = tuple(item * valid[:, :, None].to(item.dtype) for item in base)
        if self.route == "local_conv":
            local_qkv, global_qkv = self.conv_primary(base, valid), base
        elif self.route == "global_conv":
            local_qkv, global_qkv = base, self.conv_primary(base, valid)
        elif self.route == "separate_conv":
            local_qkv = self.conv_primary(base, valid)
            global_qkv = self.conv_secondary(base, valid)
        else:
            shared = self.conv_primary(base, valid)
            local_qkv = global_qkv = shared
        local_output = self._attend(local_qkv, valid, local=True)
        global_output = self._attend(global_qkv, valid, local=False)
        alpha = torch.sigmoid(self.gate(value.float())).to(value.dtype)
        fused = alpha * local_output + (1.0 - alpha) * global_output
        return self.output(fused) * valid[:, :, None].to(fused.dtype), alpha


class DualAxialBlock(nn.Module):
    def __init__(self, route: Route, dim: int = DIM, heads: int = HEADS, dropout: float = 0.1):
        super().__init__()
        self.time_norm = nn.LayerNorm(dim)
        self.time_attention = DualRangeAttention(route, dim, heads, dropout)
        self.time_drop = nn.Dropout(dropout)
        kwargs = dict(d_model=dim, nhead=heads, dim_feedforward=dim * 3, dropout=dropout,
                      activation="gelu", batch_first=True, norm_first=True)
        self.spatial = nn.TransformerEncoderLayer(**kwargs)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 3), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 3, dim))
        self.ffn_drop = nn.Dropout(dropout)

    def forward(self, value: torch.Tensor, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, channels, times, dim = value.shape
        valid = (~hidden)[:, None].expand(batch, channels, times).reshape(batch * channels, times)
        flat = value.reshape(batch * channels, times, dim)
        attended, alpha = self.time_attention(self.time_norm(flat), valid)
        flat = flat + self.time_drop(attended)
        flat = flat + self.ffn_drop(self.ffn(self.ffn_norm(flat))) * valid[:, :, None]
        value = flat.reshape(batch, channels, times, dim)
        spatial = self.spatial(value.transpose(1, 2).reshape(batch * times, channels, dim))
        value = spatial.reshape(batch, times, channels, dim).transpose(1, 2).contiguous()
        value = value.masked_fill(hidden[:, None, :, None], 0.0)
        return value, alpha.reshape(batch, channels, times, 1)


class PlainAxialBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        kwargs = dict(d_model=dim, nhead=heads, dim_feedforward=dim * 3, dropout=dropout,
                      activation="gelu", batch_first=True, norm_first=True)
        self.temporal = nn.TransformerEncoderLayer(**kwargs)
        self.spatial = nn.TransformerEncoderLayer(**kwargs)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        batch, channels, times, dim = value.shape
        value = self.temporal(value.reshape(batch * channels, times, dim)).reshape(batch, channels, times, dim)
        value = self.spatial(value.transpose(1, 2).reshape(batch * times, channels, dim))
        return value.reshape(batch, times, channels, dim).transpose(1, 2).contiguous()


class D088Encoder(nn.Module):
    def __init__(self, route: Route, dropout: float = 0.1, activation_checkpointing: bool = True):
        super().__init__()
        self.route = route
        self.activation_checkpointing = activation_checkpointing
        self.wave_patch = nn.Sequential(nn.Linear(PATCH_SAMPLES, DIM), nn.GELU(), nn.Linear(DIM, DIM))
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
        nn.init.trunc_normal_(self.channel_identity.weight, std=0.02)
        nn.init.trunc_normal_(self.time_identity.weight, std=0.02)
        nn.init.trunc_normal_(self.frequency_identity.weight, std=0.02)

    def frontend(
        self, waveform: torch.Tensor, hidden: torch.Tensor, frequency_mean: torch.Tensor, frequency_scale: torch.Tensor
    ) -> torch.Tensor:
        visible = mask_waveform(waveform, hidden)
        temporal = self.wave_patch(visible.reshape(len(waveform), CHANNELS, PATCHES, PATCH_SAMPLES))
        temporal = temporal + self.channel_identity.weight[None, :, None] + self.time_identity.weight[None, None]
        temporal = temporal.masked_fill(hidden[:, None, :, None], 0.0)
        spectral = (periodic_hann_log_psd(visible) - frequency_mean[None]) / frequency_scale[None]
        frequency = self.frequency_patch(spectral.reshape(len(waveform), CHANNELS, FREQ_PATCHES, 4))
        frequency = frequency + self.channel_identity.weight[None, :, None] + self.frequency_identity.weight[None, None]
        query = self.frequency_q_norm(temporal).reshape(len(waveform) * CHANNELS, PATCHES, DIM)
        memory = self.frequency_kv_norm(frequency).reshape(len(waveform) * CHANNELS, FREQ_PATCHES, DIM)
        fused = self.frequency_attention(query, memory, memory, need_weights=False)[0]
        value = temporal + self.frequency_drop(fused.reshape(len(waveform), CHANNELS, PATCHES, DIM))
        return value.masked_fill(hidden[:, None, :, None], 0.0)

    def forward(
        self, waveform: torch.Tensor, hidden: torch.Tensor, frequency_mean: torch.Tensor, frequency_scale: torch.Tensor,
        return_gates: bool = False,
    ):
        value = self.frontend(waveform, hidden, frequency_mean, frequency_scale)
        gates = []
        for block in self.blocks:
            if self.activation_checkpointing and self.training and torch.is_grad_enabled() and not return_gates:
                value, gate = checkpoint(block, value, hidden, use_reentrant=False, preserve_rng_state=True)
            else:
                value, gate = block(value, hidden)
            if return_gates:
                gates.append(gate)
        value = self.final_norm(value).masked_fill(hidden[:, None, :, None], 0.0)
        return (value, gates) if return_gates else value


class D088Predictor(nn.Module):
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        width = 192
        self.context = nn.Linear(DIM, width)
        self.mask_token = nn.Parameter(torch.empty(width))
        self.channel_identity = nn.Embedding(CHANNELS, width)
        self.time_identity = nn.Embedding(PATCHES, width)
        self.blocks = nn.ModuleList([PlainAxialBlock(width, 4, dropout) for _ in range(2)])
        self.norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, DIM)
        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.channel_identity.weight, std=0.02)
        nn.init.trunc_normal_(self.time_identity.weight, std=0.02)

    def forward(self, context: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        value = self.context(context)
        masked = self.mask_token[None, None, None].expand_as(value)
        value = torch.where(hidden[:, None, :, None], masked, value)
        value = value + self.channel_identity.weight[None, :, None] + self.time_identity.weight[None, None]
        for block in self.blocks:
            value = block(value)
        return self.output(self.norm(value))


class D088JEPA(nn.Module):
    def __init__(self, route: Route, dropout: float = 0.1, activation_checkpointing: bool = True):
        super().__init__()
        self.online = D088Encoder(route, dropout, activation_checkpointing)
        self.predictor = D088Predictor(dropout)
        self.target = copy.deepcopy(self.online)
        self.target.requires_grad_(False).eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.target.eval()
        return self

    def loss(self, waveform: torch.Tensor, hidden: torch.Tensor, frequency_mean: torch.Tensor, frequency_scale: torch.Tensor):
        context = self.online(waveform, hidden, frequency_mean, frequency_scale)
        predicted = self.predictor(context, hidden)
        with torch.no_grad():
            clear = torch.zeros_like(hidden)
            target = self.target(waveform, clear, frequency_mean, frequency_scale)
            target = F.layer_norm(target.float(), (target.shape[-1],))
        select = hidden[:, None, :, None].expand_as(predicted)
        loss = F.smooth_l1_loss(predicted.float()[select], target[select], beta=1.0)
        cosine = F.cosine_similarity(predicted.float()[select].reshape(-1, DIM), target[select].reshape(-1, DIM)).mean()
        return loss, {"loss": loss.detach(), "cosine": cosine.detach()}

    @torch.no_grad()
    def update_target(self, momentum: float) -> None:
        online = dict(self.online.named_parameters())
        for name, target in self.target.named_parameters():
            target.mul_(momentum).add_(online[name], alpha=1.0 - momentum)
        online_buffers = dict(self.online.named_buffers())
        for name, target in self.target.named_buffers():
            target.copy_(online_buffers[name])


class D088Classifier(nn.Module):
    def __init__(self, encoder: D088Encoder, classes: int = 80):
        super().__init__()
        self.encoder = encoder
        self.norm = nn.LayerNorm(PATCHES * DIM)
        self.head = nn.Linear(PATCHES * DIM, classes)

    def forward(self, waveform: torch.Tensor, frequency_mean: torch.Tensor, frequency_scale: torch.Tensor):
        hidden = torch.zeros((len(waveform), PATCHES), dtype=torch.bool, device=waveform.device)
        tokens = self.encoder(waveform, hidden, frequency_mean, frequency_scale)
        representation = tokens.mean(dim=1).flatten(1)
        return self.head(self.norm(representation))


def parameter_counts(model: nn.Module) -> dict[str, int]:
    return {
        "total": sum(p.numel() for p in model.parameters()),
        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }


__all__ = [
    "CHANNELS", "DIM", "PATCHES", "ROUTES", "D088JEPA", "D088Classifier",
    "periodic_hann_log_psd", "parameter_counts",
]
