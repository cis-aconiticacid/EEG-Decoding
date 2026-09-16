"""D094: compact multi-scale temporal/spatial CNN for 80-way EEG decoding.

The end-to-end model accepts the 40--440 ms crop at 1 kHz (62 x 400),
anti-alias resamples it to 250 Hz (62 x 100), and then follows the requested
multi-scale temporal -> spatial -> local temporal -> segment-statistics path.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


CHANNELS = 62
SOURCE_SAMPLES = 400
TARGET_SAMPLES = 100
TARGET_HZ = 250
TEMPORAL_KERNEL_MS = (36, 68, 132, 260)
TEMPORAL_KERNEL_SAMPLES = tuple(round(value * TARGET_HZ / 1000) for value in TEMPORAL_KERNEL_MS)
SPATIAL_FEATURES = 64
SEGMENTS = 10
SEGMENT_SAMPLES = TARGET_SAMPLES // SEGMENTS


class AntiAliasResample250(nn.Module):
    """Fixed windowed-sinc 4x decimator with an exactly 100-sample output."""

    def __init__(self, channels: int = CHANNELS, taps: int = 33, cutoff: float = 0.1125):
        super().__init__()
        if taps % 2 != 1:
            raise ValueError("taps must be odd")
        if not 0.0 < cutoff < 0.125:
            raise ValueError("cutoff must be below the 250 Hz Nyquist frequency")
        positions = torch.arange(taps, dtype=torch.float32) - taps // 2
        window = torch.hann_window(taps, periodic=False)
        kernel = 2.0 * cutoff * torch.sinc(2.0 * cutoff * positions) * window
        kernel /= kernel.sum()
        self.channels = channels
        self.taps = taps
        self.register_buffer("kernel", kernel.reshape(1, 1, taps))

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim != 3 or waveform.shape[1:] != (self.channels, SOURCE_SAMPLES):
            raise ValueError(
                f"expected [batch,{self.channels},{SOURCE_SAMPLES}], got {tuple(waveform.shape)}"
            )
        weight = self.kernel.expand(self.channels, 1, self.taps)
        output = F.conv1d(
            waveform,
            weight,
            stride=4,
            padding=self.taps // 2,
            groups=self.channels,
        )
        if output.shape[-1] != TARGET_SAMPLES:
            raise AssertionError(f"resampler produced {output.shape[-1]} samples")
        return output


class DepthwiseTemporalBlock(nn.Module):
    """Residual local temporal filtering without changing time resolution."""

    def __init__(self, features: int, kernel_size: int, dropout: float):
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd")
        self.norm = nn.GroupNorm(8, features)
        self.depthwise = nn.Conv1d(
            features,
            features,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=features,
            bias=False,
        )
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = value
        value = self.depthwise(self.norm(value))
        return residual + self.dropout(self.activation(value))


class MultiscaleTemporalSpatialCNN(nn.Module):
    """Trainable part operating on standardized 62 x 100 EEG."""

    def __init__(
        self,
        classes: int = 80,
        hidden: int = 128,
        dropout: float = 0.2,
        temporal_block_kernels: tuple[int, int] = (9, 5),
    ):
        super().__init__()
        self.temporal_branches = nn.ModuleList(
            nn.Conv1d(
                CHANNELS,
                CHANNELS,
                kernel_size=kernel,
                padding=kernel // 2,
                groups=CHANNELS,
                bias=False,
            )
            for kernel in TEMPORAL_KERNEL_SAMPLES
        )
        # Input channels are the four temporal scales.  The 62-high kernel is
        # the only pre-pooling operation that mixes electrodes.
        self.spatial = nn.Conv2d(
            len(TEMPORAL_KERNEL_SAMPLES),
            SPATIAL_FEATURES,
            kernel_size=(CHANNELS, 1),
            bias=False,
        )
        self.spatial_norm = nn.GroupNorm(8, SPATIAL_FEATURES)
        self.spatial_activation = nn.GELU()
        self.spatial_dropout = nn.Dropout(dropout)
        self.temporal_blocks = nn.ModuleList(
            DepthwiseTemporalBlock(SPATIAL_FEATURES, kernel, dropout)
            for kernel in temporal_block_kernels
        )
        statistic_features = SEGMENTS * SPATIAL_FEATURES * 2
        self.head = nn.Sequential(
            nn.LayerNorm(statistic_features),
            nn.Linear(statistic_features, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, classes),
        )

    def temporal_spatial_features(self, waveform_250hz: torch.Tensor) -> torch.Tensor:
        if waveform_250hz.ndim != 3 or waveform_250hz.shape[1:] != (CHANNELS, TARGET_SAMPLES):
            raise ValueError(
                f"expected [batch,{CHANNELS},{TARGET_SAMPLES}], got {tuple(waveform_250hz.shape)}"
            )
        scales = torch.stack([branch(waveform_250hz) for branch in self.temporal_branches], dim=1)
        value = self.spatial(scales).squeeze(2)
        value = self.spatial_dropout(self.spatial_activation(self.spatial_norm(value)))
        for block in self.temporal_blocks:
            value = block(value)
        return value

    def segment_statistics(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[1:] != (SPATIAL_FEATURES, TARGET_SAMPLES):
            raise ValueError(f"unexpected feature shape {tuple(features.shape)}")
        chunks = features.reshape(len(features), SPATIAL_FEATURES, SEGMENTS, SEGMENT_SAMPLES)
        means = chunks.mean(dim=-1)
        variances = chunks.var(dim=-1, unbiased=False)
        # Segment-major layout: [40--80 ms mean/variance, 80--120 ms, ...].
        return torch.stack((means, variances), dim=-1).permute(0, 2, 1, 3).flatten(1)

    def forward(self, waveform_250hz: torch.Tensor) -> torch.Tensor:
        features = self.temporal_spatial_features(waveform_250hz)
        return self.head(self.segment_statistics(features))


class D094Stage1Classifier(nn.Module):
    """End-to-end 1 kHz crop -> anti-aliased 250 Hz -> 80-way classifier."""

    def __init__(self, classes: int = 80, hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        self.resampler = AntiAliasResample250()
        self.classifier = MultiscaleTemporalSpatialCNN(classes=classes, hidden=hidden, dropout=dropout)
        self.register_buffer("input_mean", torch.zeros(1, CHANNELS, 1))
        self.register_buffer("input_scale", torch.ones(1, CHANNELS, 1))

    @torch.no_grad()
    def set_input_statistics(self, mean: torch.Tensor, scale: torch.Tensor) -> None:
        mean = mean.reshape(1, CHANNELS, 1).to(self.input_mean)
        scale = scale.reshape(1, CHANNELS, 1).to(self.input_scale)
        if not torch.isfinite(mean).all() or not torch.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError("input statistics must be finite with positive scale")
        self.input_mean.copy_(mean)
        self.input_scale.copy_(scale)

    def standardize_resampled(self, waveform_250hz: torch.Tensor) -> torch.Tensor:
        return (waveform_250hz - self.input_mean) / self.input_scale

    def preprocess(self, waveform_1khz: torch.Tensor) -> torch.Tensor:
        return self.standardize_resampled(self.resampler(waveform_1khz))

    def forward_resampled(self, standardized_250hz: torch.Tensor) -> torch.Tensor:
        return self.classifier(standardized_250hz)

    def forward(self, waveform_1khz: torch.Tensor) -> torch.Tensor:
        return self.forward_resampled(self.preprocess(waveform_1khz))


def parameter_counts(model: nn.Module) -> dict[str, int]:
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    }


__all__ = [
    "AntiAliasResample250",
    "CHANNELS",
    "D094Stage1Classifier",
    "DepthwiseTemporalBlock",
    "MultiscaleTemporalSpatialCNN",
    "SEGMENTS",
    "SPATIAL_FEATURES",
    "TARGET_SAMPLES",
    "TEMPORAL_KERNEL_MS",
    "TEMPORAL_KERNEL_SAMPLES",
    "parameter_counts",
]
