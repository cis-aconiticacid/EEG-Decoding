"""D057 waveform tokens to the frozen RAEv2 spatial latent (no pixel loss)."""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F
from .waveform_alignment import AxialBlock


def grid_encoding(size=16, dim=128):
    if dim % 4:
        raise ValueError("2D encoding width must be divisible by four")
    axis = torch.arange(size, dtype=torch.float32)
    omega = 1.0 / (10000 ** (torch.arange(dim // 4).float() / (dim // 4)))
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    return torch.cat([f(a.reshape(-1, 1) * omega) for a in (x, y)
                      for f in (torch.sin, torch.cos)], -1)


class QueryBlock(nn.Module):
    def __init__(self, dim=128, heads=4, dropout=.1):
        super().__init__()
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(3)])
        self.self_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 3), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(dim * 3, dim))
        self.drop = nn.Dropout(dropout)

    def forward(self, q, memory):
        x = self.norms[0](q)
        q = q + self.drop(self.self_attn(x, x, x, need_weights=False)[0])
        q = q + self.drop(self.cross_attn(self.norms[1](q), memory, memory, need_weights=False)[0])
        return q + self.drop(self.ffn(self.norms[2](q)))


class WaveformRAEEncoder(nn.Module):
    def __init__(self, coordinates, eeg_mean, eeg_scale, dim=128, depth=4,
                 heads=4, dropout=.1, query_depth=2):
        super().__init__()
        for name, value, shape in [("coordinates", coordinates, (62, 3)),
                                   ("eeg_mean", eeg_mean, (62, 1)),
                                   ("eeg_scale", eeg_scale, (62, 1))]:
            self.register_buffer(name, torch.as_tensor(value).float().reshape(shape))
        if not torch.isfinite(self.coordinates).all() or (self.eeg_scale <= 0).any():
            raise ValueError("invalid EEG geometry/statistics")
        self.patch_encoder = nn.Sequential(nn.Linear(50, dim), nn.GELU(), nn.Linear(dim, dim))
        self.electrode = nn.Embedding(62, dim)
        self.position = nn.Sequential(nn.Linear(3, 32), nn.GELU(), nn.Linear(32, dim))
        self.time_position = nn.Parameter(torch.empty(10, dim))
        self.queries = nn.Parameter(torch.empty(256, dim))
        nn.init.trunc_normal_(self.time_position, std=.02)
        nn.init.trunc_normal_(self.queries, std=.02)
        self.register_buffer("grid_position", grid_encoding(16, dim))
        self.blocks = nn.ModuleList([AxialBlock(dim, heads, dropout) for _ in range(depth)])
        self.final_norm = nn.LayerNorm(dim)
        self.query_blocks = nn.ModuleList([QueryBlock(dim, heads, dropout) for _ in range(query_depth)])
        self.projector = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 512), nn.GELU(),
                                       nn.Dropout(dropout), nn.Linear(512, 1024))

    def forward(self, raw):
        if raw.ndim != 3 or raw.shape[1:] != (62, 500):
            raise ValueError("EEG input must be [B,62,500]")
        x = ((raw - self.eeg_mean) / self.eeg_scale).reshape(-1, 62, 10, 50)
        x = self.patch_encoder(x)
        x = x + (self.electrode.weight + self.position(self.coordinates))[None, :, None]
        x = x + self.time_position[None, None]
        for block in self.blocks:
            x = block(x)
        memory = self.final_norm(x.reshape(len(raw), 620, -1))
        q = (self.queries + self.grid_position)[None].expand(len(raw), -1, -1)
        for block in self.query_blocks:
            q = block(q, memory)
        return self.projector(q)


def normalized_flat(z):
    """Dot product equals mean of the 256 same-position token cosines."""
    return F.normalize(z.float(), dim=-1, eps=1e-8).flatten(1) / math.sqrt(z.shape[1])


def latent_loss(predicted, targets, image_ids, temperature=.07, contrastive_weight=.1):
    if predicted.shape != targets.shape or predicted.ndim != 3:
        raise ValueError("prediction and target must share [B,P,D] shape")
    predicted, targets = predicted.float(), targets.detach().float()
    mse = (predicted - targets).square().mean()
    unique, inverse = torch.unique(image_ids, sorted=True, return_inverse=True)
    if len(unique) < 2:
        contrastive = predicted.sum() * 0.0
    else:
        # Every repeated image uses precisely the same frozen target row.
        first = torch.stack([torch.nonzero(inverse == j, as_tuple=False)[0, 0]
                             for j in range(len(unique))])
        logits = normalized_flat(predicted) @ normalized_flat(targets[first]).T / temperature
        forward = F.cross_entropy(logits, inverse)
        positive = inverse[:, None].eq(torch.arange(len(unique), device=inverse.device)[None])
        reverse = (torch.logsumexp(logits, dim=0)
                   - torch.logsumexp(logits.masked_fill(~positive, -torch.inf), dim=0)).mean()
        contrastive = (forward + reverse) * .5
    loss = mse + contrastive_weight * contrastive
    return loss, {"latent_mse": mse.detach(), "contrastive": contrastive.detach(),
                  "loss": loss.detach(), "unique_images": len(unique)}


def learning_rate(step, steps_per_epoch=225):
    """One-based optimizer update, exact endpoints of approved schedule."""
    warm, end = 3 * steps_per_epoch, 150 * steps_per_epoch
    if step <= warm:
        return 3e-5 + (3e-4 - 3e-5) * (step - 1) / (warm - 1)
    phase = min(1., (step - warm) / (end - warm))
    return 3e-5 + .5 * (3e-4 - 3e-5) * (1 + math.cos(math.pi * phase))
