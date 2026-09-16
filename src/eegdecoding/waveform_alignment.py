"""D056: retain 62 electrodes x 10 waveform patches until final pooling."""
from __future__ import annotations

import torch
from torch import nn


class AxialBlock(nn.Module):
    def __init__(self, dim=128, heads=4, dropout=0.1):
        super().__init__()
        kwargs = dict(d_model=dim, nhead=heads, dim_feedforward=dim * 3,
                      dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.temporal = nn.TransformerEncoderLayer(**kwargs)
        self.spatial = nn.TransformerEncoderLayer(**kwargs)

    def forward(self, x):
        b, c, t, d = x.shape
        x = self.temporal(x.reshape(b * c, t, d)).reshape(b, c, t, d)
        x = self.spatial(x.transpose(1, 2).reshape(b * t, c, d))
        return x.reshape(b, t, c, d).transpose(1, 2).contiguous()


class WaveformImageEncoder(nn.Module):
    def __init__(self, coordinates, eeg_mean, eeg_scale, target_mean, target_rms,
                 dim=128, depth=4, heads=4, dropout=0.1):
        super().__init__()
        self.register_buffer("coordinates", torch.as_tensor(coordinates).float().reshape(62, 3))
        self.register_buffer("eeg_mean", torch.as_tensor(eeg_mean).float().reshape(62, 1))
        self.register_buffer("eeg_scale", torch.as_tensor(eeg_scale).float().reshape(62, 1))
        self.register_buffer("target_mean", torch.as_tensor(target_mean).float().reshape(1024))
        self.register_buffer("target_rms", torch.as_tensor(target_rms).float().reshape(()))
        if (self.eeg_scale <= 0).any() or self.target_rms <= 0:
            raise ValueError("normalization scales must be positive")
        self.patch_encoder = nn.Sequential(nn.Linear(50, dim), nn.GELU(), nn.Linear(dim, dim))
        self.electrode = nn.Embedding(62, dim)
        self.position = nn.Sequential(nn.Linear(3, 32), nn.GELU(), nn.Linear(32, dim))
        self.time_position = nn.Parameter(torch.empty(10, dim))
        self.query = nn.Parameter(torch.empty(1, 1, dim))
        nn.init.trunc_normal_(self.time_position, std=0.02)
        nn.init.trunc_normal_(self.query, std=0.02)
        self.blocks = nn.ModuleList([AxialBlock(dim, heads, dropout) for _ in range(depth)])
        self.final_norm = nn.LayerNorm(dim)
        self.pool = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.projector = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 512), nn.GELU(),
                                       nn.Dropout(dropout), nn.Linear(512, 1024))

    def encode_tokens(self, raw):
        if raw.ndim != 3 or raw.shape[1:] != (62, 500):
            raise ValueError(f"expected [B,62,500], got {tuple(raw.shape)}")
        standardized = (raw - self.eeg_mean) / self.eeg_scale
        patches = standardized.reshape(len(raw), 62, 10, 50)
        x = self.patch_encoder(patches)
        x = x + (self.electrode.weight + self.position(self.coordinates))[None, :, None, :]
        x = x + self.time_position[None, None]
        for block in self.blocks:
            x = block(x)
        return self.final_norm(x.reshape(len(raw), 620, -1))

    def forward(self, raw):
        tokens = self.encode_tokens(raw)
        pooled, _ = self.pool(self.query.expand(len(raw), -1, -1), tokens, tokens, need_weights=False)
        return self.target_mean + self.target_rms * self.projector(pooled[:, 0])
