"""D095: multi-scale waveform/frequency axial classifier for EEG-ImageNet.

The model follows the subject-0 80-way architecture agreed for the 40--440 ms
crop.  It keeps the 62 electrodes separate until the explicitly spatial
operations, produces 16 aligned 25 ms tokens, and uses the 97 ms local STFT
view only through a learnable residual gate.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


CHANNELS = 62
SAMPLES = 400
SAMPLE_RATE = 1000
PATCHES = 16
PATCH_SAMPLES = 25
MULTISCALE_KERNELS = (9, 17, 33)
BRANCH_FEATURES = 8
CONCAT_FEATURES = len(MULTISCALE_KERNELS) * BRANCH_FEATURES
FUSED_FEATURES = 32
DIM = 192
FRONTEND_DIM = DIM
DEPTH = 4
HEADS = 6
FFN_DIM = 576
SPATIAL_BLOCKS = (2, 4)
FREQUENCY_WINDOW_SAMPLES = 97
FFT_SIZE = 256
FREQUENCY_LOW_HZ = 12.0
FREQUENCY_HIGH_HZ = 80.0


def frequency_bin_centres() -> torch.Tensor:
    """Return the 256-point FFT bin centres retained by the 12--80 Hz mask."""

    bins = torch.fft.rfftfreq(FFT_SIZE, d=1.0 / SAMPLE_RATE)
    return bins[(bins >= FREQUENCY_LOW_HZ) & (bins <= FREQUENCY_HIGH_HZ)]


FREQUENCY_BINS = int(frequency_bin_centres().numel())


def normalize_unit_sphere_coordinates(coordinates: torch.Tensor) -> torch.Tensor:
    """Normalize each montage coordinate to unit radius without changing axes."""

    coordinates = torch.as_tensor(coordinates, dtype=torch.float32)
    if coordinates.shape != (CHANNELS, 3) or not torch.isfinite(coordinates).all():
        raise ValueError(f"coordinates must be finite [{CHANNELS},3]")
    radius = coordinates.norm(dim=-1, keepdim=True)
    if (radius <= 0).any():
        raise ValueError("montage coordinates must have non-zero radius")
    return coordinates / radius


def aligned_hann_log_power(standardized_waveform: torch.Tensor) -> torch.Tensor:
    """Compute 16 centre-aligned 97 ms Hann/FFT log-power tokens.

    Input is ``[B,62,400]`` at 1 kHz.  Token centres are samples
    12, 37, ..., 387, matching the centres of the sixteen non-overlapping
    25-sample waveform patches.  Reflection padding follows the usual centred
    STFT convention at the two crop boundaries.
    """

    if standardized_waveform.ndim != 3 or standardized_waveform.shape[1:] != (CHANNELS, SAMPLES):
        raise ValueError(
            f"expected [batch,{CHANNELS},{SAMPLES}], got {tuple(standardized_waveform.shape)}"
        )
    half_window = FREQUENCY_WINDOW_SAMPLES // 2
    padded = F.pad(standardized_waveform.float(), (half_window, half_window), mode="reflect")
    # Starting at padded sample 12 makes the first window's original-time
    # centre sample 12; the stride then follows the 25-sample waveform tokens.
    windows = padded[..., PATCH_SAMPLES // 2 :].unfold(
        dimension=-1,
        size=FREQUENCY_WINDOW_SAMPLES,
        step=PATCH_SAMPLES,
    )
    if windows.shape[-2:] != (PATCHES, FREQUENCY_WINDOW_SAMPLES):
        raise AssertionError(f"unexpected aligned-window shape {tuple(windows.shape)}")
    hann = torch.hann_window(
        FREQUENCY_WINDOW_SAMPLES,
        periodic=True,
        dtype=windows.dtype,
        device=windows.device,
    )
    transformed = torch.fft.rfft(windows * hann, n=FFT_SIZE, dim=-1)
    frequencies = torch.fft.rfftfreq(FFT_SIZE, d=1.0 / SAMPLE_RATE, device=windows.device)
    retained = (frequencies >= FREQUENCY_LOW_HZ) & (frequencies <= FREQUENCY_HIGH_HZ)
    return torch.log1p(transformed.abs().square()[..., retained])


class PerElectrodeMultiscaleFrontend(nn.Module):
    """Three independent per-electrode temporal scales and 24 -> 32 fusion."""

    def __init__(self):
        super().__init__()
        self.branches = nn.ModuleList(
            nn.Conv1d(
                CHANNELS,
                CHANNELS * BRANCH_FEATURES,
                kernel_size=kernel,
                padding=kernel // 2,
                groups=CHANNELS,
                bias=False,
            )
            for kernel in MULTISCALE_KERNELS
        )
        self.activation = nn.GELU()
        self.fusion = nn.Conv1d(
            CHANNELS * CONCAT_FEATURES,
            CHANNELS * FUSED_FEATURES,
            kernel_size=1,
            groups=CHANNELS,
            bias=True,
        )
        self.patch_projection = nn.Linear(FUSED_FEATURES, FRONTEND_DIM)

    def multiscale_features(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim != 3 or waveform.shape[1:] != (CHANNELS, SAMPLES):
            raise ValueError(f"expected [batch,{CHANNELS},{SAMPLES}], got {tuple(waveform.shape)}")
        batch = len(waveform)
        branches = [
            self.activation(branch(waveform)).reshape(batch, CHANNELS, BRANCH_FEATURES, SAMPLES)
            for branch in self.branches
        ]
        concatenated = torch.cat(branches, dim=2)
        fused = self.fusion(concatenated.reshape(batch, CHANNELS * CONCAT_FEATURES, SAMPLES))
        return self.activation(fused).reshape(batch, CHANNELS, FUSED_FEATURES, SAMPLES)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        features = self.multiscale_features(waveform)
        # The temporal convolutions already encode local morphology.  Average
        # within each exact 25 ms patch to obtain the requested 32-vector, then
        # apply the 32 -> 192 patch embedding.
        patches = features.reshape(
            len(features), CHANNELS, FUSED_FEATURES, PATCHES, PATCH_SAMPLES
        ).mean(dim=-1)
        patches = patches.permute(0, 1, 3, 2).contiguous()
        return self.patch_projection(patches)


class WaveformFrequencyGate(nn.Module):
    """Scalar residual frequency gate for each electrode/time token."""

    def __init__(self):
        super().__init__()
        self.waveform_norm = nn.LayerNorm(FRONTEND_DIM)
        self.frequency_norm = nn.LayerNorm(FRONTEND_DIM)
        self.gate = nn.Linear(2 * FRONTEND_DIM, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, math.log(0.1 / 0.9))

    def forward(
        self,
        waveform_tokens: torch.Tensor,
        frequency_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if waveform_tokens.shape != frequency_tokens.shape or waveform_tokens.shape[-1] != FRONTEND_DIM:
            raise ValueError(f"waveform and frequency tokens must have the same [...,{FRONTEND_DIM}] shape")
        gate_input = torch.cat(
            (self.waveform_norm(waveform_tokens), self.frequency_norm(frequency_tokens)), dim=-1
        )
        gate = torch.sigmoid(self.gate(gate_input))
        return waveform_tokens + gate * frequency_tokens, gate


class TemporalSpatialBlock(nn.Module):
    """Local temporal convolution, global temporal attention, optional spatial attention, and FFN."""

    def __init__(self, spatial_attention: bool, dropout: float):
        super().__init__()
        self.has_spatial_attention = bool(spatial_attention)
        self.local_norm = nn.LayerNorm(DIM)
        self.local_conv = nn.Conv1d(DIM, DIM, kernel_size=3, padding=1, groups=DIM, bias=True)
        self.local_drop = nn.Dropout(dropout)
        self.temporal_norm = nn.LayerNorm(DIM)
        self.temporal_attention = nn.MultiheadAttention(
            DIM, HEADS, dropout=dropout, batch_first=True
        )
        self.temporal_drop = nn.Dropout(dropout)
        if self.has_spatial_attention:
            self.spatial_norm = nn.LayerNorm(DIM)
            self.spatial_attention = nn.MultiheadAttention(
                DIM, HEADS, dropout=dropout, batch_first=True
            )
            self.spatial_drop = nn.Dropout(dropout)
        else:
            self.spatial_norm = None
            self.spatial_attention = None
            self.spatial_drop = None
        self.ffn_norm = nn.LayerNorm(DIM)
        self.ffn = nn.Sequential(
            nn.Linear(DIM, FFN_DIM),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(FFN_DIM, DIM),
        )
        self.ffn_drop = nn.Dropout(dropout)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 4 or value.shape[1:] != (CHANNELS, PATCHES, DIM):
            raise ValueError(f"expected [batch,{CHANNELS},{PATCHES},{DIM}], got {tuple(value.shape)}")
        batch = len(value)

        local = self.local_norm(value).reshape(batch * CHANNELS, PATCHES, DIM).transpose(1, 2)
        local = self.local_conv(local).transpose(1, 2).reshape(batch, CHANNELS, PATCHES, DIM)
        value = value + self.local_drop(F.gelu(local))

        temporal = self.temporal_norm(value).reshape(batch * CHANNELS, PATCHES, DIM)
        temporal = self.temporal_attention(temporal, temporal, temporal, need_weights=False)[0]
        value = value + self.temporal_drop(temporal.reshape(batch, CHANNELS, PATCHES, DIM))

        if self.has_spatial_attention:
            spatial = self.spatial_norm(value).transpose(1, 2).reshape(batch * PATCHES, CHANNELS, DIM)
            spatial = self.spatial_attention(spatial, spatial, spatial, need_weights=False)[0]
            spatial = spatial.reshape(batch, PATCHES, CHANNELS, DIM).transpose(1, 2).contiguous()
            value = value + self.spatial_drop(spatial)

        return value + self.ffn_drop(self.ffn(self.ffn_norm(value)))


class LearnedSpatialPooling(nn.Module):
    """Learn one normalized electrode weight distribution per time token."""

    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(DIM)
        self.score = nn.Linear(DIM, 1)

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if value.ndim != 4 or value.shape[1:] != (CHANNELS, PATCHES, DIM):
            raise ValueError(f"expected [batch,{CHANNELS},{PATCHES},{DIM}], got {tuple(value.shape)}")
        weights = torch.softmax(self.score(self.norm(value)).float(), dim=1)
        pooled = (value.float() * weights).sum(dim=1).to(value.dtype)
        return pooled, weights


class D095HybridClassifier(nn.Module):
    """End-to-end standardized 62 x 400 waveform to 80-class logits."""

    def __init__(
        self,
        coordinates: torch.Tensor,
        classes: int = 80,
        dropout: float = 0.1,
        head_dropout: float = 0.2,
        use_frequency: bool = True,
        excluded_band_hz: tuple[float, float] | None = None,
        post_fusion_projection: bool = False,
    ):
        super().__init__()
        self.classes = int(classes)
        self.use_frequency = bool(use_frequency)
        self.excluded_band_hz = excluded_band_hz
        self.post_fusion_projection_enabled = bool(post_fusion_projection)
        if excluded_band_hz is not None and not (0 < excluded_band_hz[0] < excluded_band_hz[1] < SAMPLE_RATE / 2):
            raise ValueError('excluded_band_hz must be inside (0, Nyquist)')
        self.frontend = PerElectrodeMultiscaleFrontend()
        if self.use_frequency:
            self.frequency_projection: nn.Linear | None = nn.Linear(FREQUENCY_BINS, FRONTEND_DIM)
            self.frequency_gate: WaveformFrequencyGate | None = WaveformFrequencyGate()
        else:
            self.frequency_projection = None
            self.frequency_gate = None
        self.electrode_embedding = nn.Embedding(CHANNELS, FRONTEND_DIM)
        self.coordinate_embedding = nn.Sequential(
            nn.Linear(3, 32), nn.GELU(), nn.Linear(32, FRONTEND_DIM)
        )
        self.temporal_embedding = nn.Embedding(PATCHES, FRONTEND_DIM)
        self.input_dropout = nn.Dropout(dropout)
        self.post_fusion_projection = (
            nn.Linear(DIM, DIM) if self.post_fusion_projection_enabled else nn.Identity()
        )
        self.blocks = nn.ModuleList(
            [
                TemporalSpatialBlock(index in SPATIAL_BLOCKS, dropout)
                for index in range(1, DEPTH + 1)
            ]
        )
        self.spatial_pooling = LearnedSpatialPooling()
        self.head_norm = nn.LayerNorm(PATCHES * DIM)
        self.head_dropout = nn.Dropout(head_dropout)
        self.classifier = nn.Linear(PATCHES * DIM, self.classes)
        self.register_buffer("coordinates", normalize_unit_sphere_coordinates(coordinates))
        self.register_buffer("input_mean", torch.zeros(1, CHANNELS, 1))
        self.register_buffer("input_scale", torch.ones(1, CHANNELS, 1))

        nn.init.trunc_normal_(self.electrode_embedding.weight, std=0.02)
        nn.init.trunc_normal_(self.temporal_embedding.weight, std=0.02)

    @torch.no_grad()
    def set_input_statistics(self, mean: torch.Tensor, scale: torch.Tensor) -> None:
        mean = torch.as_tensor(mean).reshape(1, CHANNELS, 1).to(self.input_mean)
        scale = torch.as_tensor(scale).reshape(1, CHANNELS, 1).to(self.input_scale)
        if not torch.isfinite(mean).all() or not torch.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError("input statistics must be finite with positive scale")
        self.input_mean.copy_(mean)
        self.input_scale.copy_(scale)

    def filter_input(self, waveform: torch.Tensor) -> torch.Tensor:
        """Trial-local Fourier notch on the 400-sample crop, before both branches.

        This removes discrete DFT components, not an ideal continuous-frequency
        interval. At 1kHz/400 samples, 18--28Hz removes 20/22.5/25/27.5Hz.
        """
        if self.excluded_band_hz is None:
            return waveform
        frequencies = torch.fft.rfftfreq(waveform.shape[-1], 1 / SAMPLE_RATE, device=waveform.device)
        low, high = self.excluded_band_hz
        spectrum = torch.fft.rfft(waveform.float(), dim=-1)
        spectrum[..., (frequencies >= low) & (frequencies <= high)] = 0
        return torch.fft.irfft(spectrum, n=waveform.shape[-1], dim=-1)

    def standardize(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim != 3 or waveform.shape[1:] != (CHANNELS, SAMPLES):
            raise ValueError(f"expected [batch,{CHANNELS},{SAMPLES}], got {tuple(waveform.shape)}")
        return (self.filter_input(waveform) - self.input_mean) / self.input_scale

    def tokens(
        self,
        standardized_waveform: torch.Tensor,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor | None]]:
        waveform_tokens = self.frontend(standardized_waveform)
        frequency_tokens = None
        gate = None
        if self.use_frequency:
            if self.frequency_projection is None or self.frequency_gate is None:
                raise RuntimeError("frequency branch is enabled but its modules are missing")
            frequency_features = aligned_hann_log_power(standardized_waveform)
            if self.excluded_band_hz is not None:
                low, high = self.excluded_band_hz
                bins = frequency_bin_centres().to(frequency_features.device)
                frequency_features = frequency_features.masked_fill((bins >= low) & (bins <= high), 0)
            frequency_tokens = self.frequency_projection(frequency_features)
            fused, gate = self.frequency_gate(waveform_tokens, frequency_tokens)
        else:
            fused = waveform_tokens

        position = (
            self.electrode_embedding.weight[:, None, :]
            + self.coordinate_embedding(self.coordinates)[:, None, :]
            + self.temporal_embedding.weight[None, :, :]
        )
        value = self.post_fusion_projection(self.input_dropout(fused + position[None]))
        for block in self.blocks:
            value = block(value)
        if return_aux:
            return value, {
                "waveform_tokens": waveform_tokens,
                "frequency_tokens": frequency_tokens,
                "frequency_gate": gate,
            }
        return value

    def forward_standardized(
        self,
        standardized_waveform: torch.Tensor,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor | None]]:
        token_result = self.tokens(standardized_waveform, return_aux=return_aux)
        if return_aux:
            tokens, auxiliary = token_result
        else:
            tokens = token_result
            auxiliary = None
        pooled, spatial_weights = self.spatial_pooling(tokens)
        flattened = pooled.flatten(1)
        logits = self.classifier(self.head_dropout(self.head_norm(flattened)))
        if return_aux:
            auxiliary = {**auxiliary, "spatial_pooling_weights": spatial_weights}
            return logits, auxiliary
        return logits

    def forward(
        self,
        waveform: torch.Tensor,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor | None]]:
        return self.forward_standardized(self.standardize(waveform), return_aux=return_aux)


def parameter_counts(model: nn.Module) -> dict[str, int]:
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    }


__all__ = [
    "BRANCH_FEATURES",
    "CHANNELS",
    "CONCAT_FEATURES",
    "DEPTH",
    "DIM",
    "D095HybridClassifier",
    "FFT_SIZE",
    "FFN_DIM",
    "FREQUENCY_BINS",
    "FREQUENCY_WINDOW_SAMPLES",
    "FRONTEND_DIM",
    "FUSED_FEATURES",
    "HEADS",
    "MULTISCALE_KERNELS",
    "PATCHES",
    "PATCH_SAMPLES",
    "SPATIAL_BLOCKS",
    "aligned_hann_log_power",
    "frequency_bin_centres",
    "normalize_unit_sphere_coordinates",
    "parameter_counts",
]
