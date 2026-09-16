"""Pure 80-way Stage-1 classifier on top of the D089 EEG encoder.

No image embedding, image query, latent regression, retrieval objective, or
JEPA target network is present in this module.
"""

from __future__ import annotations

import math
import types

import torch
from torch import nn
from torch.nn import functional as F

from d089_masked_retrieval import CHANNELS, DIM, PATCHES, D089Encoder, apply_rope


def _sdpa_attend(self, qkv: tuple[torch.Tensor, torch.Tensor, torch.Tensor], local: bool) -> torch.Tensor:
    """State-dict-compatible replacement for explicit temporal attention."""

    q, k, v = (self._heads(item) for item in qkv)
    positions = torch.arange(q.shape[-2], device=q.device)
    q, k = apply_rope(q, positions), apply_rope(k, positions)
    attention_mask = None
    if local:
        distance = (positions[:, None] - positions[None, :]).abs()
        attention_mask = distance <= 1
    result = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=attention_mask,
        dropout_p=self.dropout if self.training else 0.0,
        is_causal=False,
        scale=1.0 / math.sqrt(self.head_dim),
    )
    result = result.permute(0, 2, 1, 3)
    return result.reshape(len(q), q.shape[-2], self.dim)


def enable_sdpa_temporal_attention(encoder: D089Encoder) -> int:
    count = 0
    for block in encoder.blocks:
        attention = block.time_attention
        attention._attend = types.MethodType(_sdpa_attend, attention)
        count += 1
    return count


class Stage1Classifier(nn.Module):
    """Keep channel/time tokens through the encoder and fuse channels at output."""

    def __init__(self, encoder: D089Encoder, classes: int = 80, head_dropout: float = 0.2):
        super().__init__()
        self.encoder = encoder
        self.norm = nn.LayerNorm(PATCHES * DIM)
        self.dropout = nn.Dropout(head_dropout)
        self.head = nn.Linear(PATCHES * DIM, classes)

    def forward(self, waveform: torch.Tensor, frequency_mean: torch.Tensor, frequency_scale: torch.Tensor):
        hidden = torch.zeros((len(waveform), PATCHES), dtype=torch.bool, device=waveform.device)
        tokens = self.encoder(waveform, hidden, frequency_mean, frequency_scale)
        representation = tokens.mean(dim=1).flatten(1)
        return self.head(self.dropout(self.norm(representation)))


def parameter_counts(model: nn.Module) -> dict[str, int]:
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    }


__all__ = ["Stage1Classifier", "enable_sdpa_temporal_attention", "parameter_counts"]
