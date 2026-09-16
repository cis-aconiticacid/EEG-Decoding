"""Sample-efficient EEG to full RAEv2-DINOv3 spatial-latent prediction."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from d089_masked_retrieval import (
    CHANNELS,
    DIM,
    HEADS,
    IMAGE_DIM,
    IMAGE_TOKENS,
    PATCHES,
    D089Encoder,
    QueryBlock,
    fixed_2d_sincos,
    normalized_flat,
)


class SampleEfficientLatentPredictor(nn.Module):
    """Shared query decoder; no per-image or category-specific parameters."""

    def __init__(self, encoder: D089Encoder, query_depth: int = 4, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        self.query = nn.Parameter(torch.empty(IMAGE_TOKENS, DIM))
        self.register_buffer("query_position", fixed_2d_sincos(16, DIM), persistent=True)
        self.blocks = nn.ModuleList([QueryBlock(DIM, HEADS, dropout) for _ in range(query_depth)])
        self.norm = nn.LayerNorm(DIM)
        self.output = nn.Sequential(
            nn.Linear(DIM, 512), nn.GELU(), nn.Dropout(dropout), nn.Linear(512, IMAGE_DIM)
        )
        nn.init.normal_(self.query, std=0.02)

    def forward(
        self,
        waveform: torch.Tensor,
        frequency_mean: torch.Tensor,
        frequency_scale: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if hidden is None:
            hidden = torch.zeros((len(waveform), PATCHES), dtype=torch.bool, device=waveform.device)
        memory = self.encoder(waveform, hidden, frequency_mean, frequency_scale)
        memory = memory.reshape(len(waveform), CHANNELS * PATCHES, DIM)
        query = (self.query + self.query_position)[None].expand(len(waveform), -1, -1)
        for block in self.blocks:
            if self.training and torch.is_grad_enabled():
                query = checkpoint(block, query, memory, use_reentrant=False, preserve_rng_state=True)
            else:
                query = block(query, memory)
        return self.output(self.norm(query))


def latent_objective(
    first: torch.Tensor,
    second: torch.Tensor,
    target: torch.Tensor,
    negatives: torch.Tensor,
    temperature: float = 0.07,
    cosine_weight: float = 0.2,
    contrastive_weight: float = 0.1,
    consistency_weight: float = 0.05,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Full-latent regression plus train-only negatives and view consistency."""
    target = target.detach().float()
    negatives = negatives.detach().float()
    first_float, second_float = first.float(), second.float()
    regression = 0.5 * (
        F.smooth_l1_loss(first_float, target, beta=1.0)
        + F.smooth_l1_loss(second_float, target, beta=1.0)
    )
    token_cosine = 1.0 - 0.5 * (
        F.cosine_similarity(first_float, target, dim=-1).mean()
        + F.cosine_similarity(second_float, target, dim=-1).mean()
    )
    candidates = torch.cat((target, negatives), dim=0)
    candidate_flat = normalized_flat(candidates)
    labels = torch.arange(len(target), device=target.device)
    first_logits = normalized_flat(first_float) @ candidate_flat.T / temperature
    second_logits = normalized_flat(second_float) @ candidate_flat.T / temperature
    contrastive = 0.5 * (
        F.cross_entropy(first_logits, labels) + F.cross_entropy(second_logits, labels)
    )
    consistency = 1.0 - F.cosine_similarity(
        normalized_flat(first_float), normalized_flat(second_float), dim=-1
    ).mean()
    total = (
        regression
        + cosine_weight * token_cosine
        + contrastive_weight * contrastive
        + consistency_weight * consistency
    )
    return total, {
        "regression": regression.detach(),
        "token_cosine": token_cosine.detach(),
        "contrastive": contrastive.detach(),
        "consistency": consistency.detach(),
    }


def trainable_parameter_counts(model: nn.Module) -> dict[str, int]:
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "encoder_trainable": sum(
            parameter.numel() for parameter in model.encoder.parameters() if parameter.requires_grad
        ),
    }


__all__ = ["SampleEfficientLatentPredictor", "latent_objective", "trainable_parameter_counts"]
