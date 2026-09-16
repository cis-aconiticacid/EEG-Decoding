"""Measure temporal and frequency-branch magnitudes before choosing fusion scale."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from d089_masked_retrieval import (  # noqa: E402
    CHANNELS,
    DIM,
    FREQ_PATCHES,
    PATCHES,
    PATCH_SAMPLES,
    D089Encoder,
    MaskedWaveformReconstructor,
    masked_waveform,
    periodic_hann_log_psd,
)
import run_d089_corrected_masked_retrieval as common  # noqa: E402
from run_d089_local_curve_pretrain import cpu_statistics, frequency_statistics  # noqa: E402


def magnitude(value: torch.Tensor) -> dict[str, float]:
    flat = value.float().flatten()
    absolute = flat.abs()
    return {
        "mean": float(flat.mean()),
        "std": float(flat.std()),
        "rms": float(flat.square().mean().sqrt()),
        "abs_p50": float(torch.quantile(absolute, 0.50)),
        "abs_p90": float(torch.quantile(absolute, 0.90)),
        "abs_p99": float(torch.quantile(absolute, 0.99)),
        "abs_max": float(absolute.max()),
        "l2_per_sample_mean": float(value.float().flatten(1).norm(dim=1).mean()),
    }


def main() -> None:
    cfg = json.loads((ROOT / "config/d089_local_curve_pretrain.json").read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = common.load_subject_data(cfg)
    mean, scale, fit_waveform = cpu_statistics(data["waveform"], data["fit"])
    validation = (
        (data["waveform"][data["validation"][:8]] - mean[None, :, None]) / scale[None, :, None]
    ).contiguous()
    frequency_mean, frequency_scale = frequency_statistics(fit_waveform)
    hidden = common.make_masks(
        len(validation), 0.50, torch.Generator().manual_seed(8913), torch.device("cpu")
    )

    checkpoint = torch.load(
        ROOT / "runs/d089-local-curve-pretrain/local_conv/best.pt",
        map_location="cpu",
        weights_only=False,
    )
    model = MaskedWaveformReconstructor(D089Encoder("local_conv", dropout=cfg["dropout"]))
    model.load_state_dict(checkpoint["model"])
    encoder = model.encoder.to(device).eval()
    waveform = validation.to(device)
    hidden = hidden.to(device)
    frequency_mean = frequency_mean.to(device)
    frequency_scale = frequency_scale.to(device)

    with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        visible = masked_waveform(waveform, hidden)
        patches = visible.reshape(len(waveform), CHANNELS, PATCHES, PATCH_SAMPLES)
        content = encoder.wave_patch(patches)
        identities = encoder.channel_identity.weight[None, :, None] + encoder.time_identity.weight[None, None]
        clear = content + identities
        masked = encoder.mask_token[None, None, None] + identities
        temporal = torch.where(hidden[:, None, :, None], masked, clear)

        spectral = (periodic_hann_log_psd(visible) - frequency_mean[None]) / frequency_scale[None]
        frequency_tokens = encoder.frequency_patch(
            spectral.reshape(len(waveform), CHANNELS, FREQ_PATCHES, 4)
        )
        frequency_tokens = (
            frequency_tokens
            + encoder.channel_identity.weight[None, :, None]
            + encoder.frequency_identity.weight[None, None]
        )
        query = encoder.frequency_q_norm(temporal).reshape(len(waveform) * CHANNELS, PATCHES, DIM)
        memory = encoder.frequency_kv_norm(frequency_tokens).reshape(
            len(waveform) * CHANNELS, FREQ_PATCHES, DIM
        )
        frequency_delta = encoder.frequency_attention(query, memory, memory, need_weights=False)[0]
        frequency_delta = frequency_delta.reshape(len(waveform), CHANNELS, PATCHES, DIM)

    temporal_f = temporal.float()
    delta_f = frequency_delta.float()
    temporal_norm = temporal_f.flatten(1).norm(dim=1)
    delta_norm = delta_f.flatten(1).norm(dim=1)
    cosine = torch.nn.functional.cosine_similarity(temporal_f.flatten(1), delta_f.flatten(1), dim=1)
    per_sample = []
    for index in range(len(waveform)):
        item = {
            "sample": index,
            "temporal_rms": float(temporal_f[index].square().mean().sqrt()),
            "frequency_delta_rms": float(delta_f[index].square().mean().sqrt()),
            "frequency_to_temporal_l2": float(delta_norm[index] / temporal_norm[index]),
            "cosine": float(cosine[index]),
        }
        for fusion_scale in (1.0, 0.2):
            fused = temporal_f[index] + fusion_scale * delta_f[index]
            item[f"scale_{fusion_scale:g}_increment_over_fused_l2"] = float(
                (fusion_scale * delta_norm[index]) / fused.flatten().norm()
            )
        per_sample.append(item)

    report = {
        "checkpoint": {
            "route": checkpoint["route"],
            "step": checkpoint["step"],
            "validation": checkpoint["validation"],
        },
        "samples": len(waveform),
        "mask_ratio": 0.50,
        "temporal": magnitude(temporal_f),
        "frequency_delta_before_scaling": magnitude(delta_f),
        "aggregate": {
            "frequency_to_temporal_rms": float(delta_f.square().mean().sqrt() / temporal_f.square().mean().sqrt()),
            "frequency_to_temporal_l2_mean": float((delta_norm / temporal_norm).mean()),
            "cosine_mean": float(cosine.mean()),
        },
        "per_sample": per_sample,
    }
    out = ROOT / "reports/d089_frequency_fusion_magnitude.json"
    common.atomic_json(out, report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
