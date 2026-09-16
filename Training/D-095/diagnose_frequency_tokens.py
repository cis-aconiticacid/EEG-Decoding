"""Verify that aligned frequency tokens differ across the 16 time positions."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
EXPERIMENT_DIR = Path(__file__).resolve().parent
ROOT = EXPERIMENT_DIR.parents[1]
sys.path.insert(0, str(ROOT))

from model import (  # noqa: E402
    CHANNELS,
    FREQUENCY_WINDOW_SAMPLES,
    PATCHES,
    PATCH_SAMPLES,
    aligned_hann_log_power,
    frequency_bin_centres,
)
from eegdecoding.data import load_part  # noqa: E402


def main() -> int:
    config = json.loads((EXPERIMENT_DIR / "config/config.json").read_text(encoding="utf-8"))
    metadata_path = ROOT / config["metadata"]
    waveform_path = ROOT / config["waveform"]
    if metadata_path.is_file() and waveform_path.is_file():
        metadata = np.load(metadata_path, allow_pickle=False)
        waveform = np.load(waveform_path, mmap_mode="r", allow_pickle=False)
        subject_indices = np.flatnonzero(metadata["subject"] == config["subject"])[:32]
        samples = torch.from_numpy(np.asarray(waveform[subject_indices], dtype=np.float32).copy())
        source_kind = "prepared_npy"
    else:
        archive = load_part(ROOT / config["archive"])
        rows = [
            row for row in archive["dataset"]
            if int(row["subject"]) == config["subject"]
        ][:32]
        start, stop = config["crop"]
        samples = torch.stack([row["eeg_data"][:, start:stop].float() for row in rows])
        source_kind = "pth_archive"

    checkpoint = torch.load(
        EXPERIMENT_DIR / config["run_dir"] / "latest.pt",
        map_location="cpu",
        weights_only=True,
    )
    mean = checkpoint["input_mean"].reshape(1, CHANNELS, 1)
    scale = checkpoint["input_scale"].reshape(1, CHANNELS, 1)
    standardized = (samples - mean) / scale
    features = aligned_hann_log_power(standardized)
    model_state = checkpoint["model"]
    frequency_tokens = F.linear(
        features,
        model_state["frequency_projection.weight"],
        model_state["frequency_projection.bias"],
    )

    def comparison(value: torch.Tensor) -> dict[str, object]:
        adjacent_left = value[:, :, :-1]
        adjacent_right = value[:, :, 1:]
        exact_equal = (adjacent_left == adjacent_right).all(dim=-1)
        return {
            "shape": list(value.shape),
            "adjacent_pair_mean_absolute_difference": [
                float(item)
                for item in (adjacent_right - adjacent_left).abs().mean(dim=(0, 1, 3))
            ],
            "adjacent_pair_mean_cosine_similarity": [
                float(item)
                for item in F.cosine_similarity(adjacent_left, adjacent_right, dim=-1).mean(dim=(0, 1))
            ],
            "exactly_equal_adjacent_token_pairs": int(exact_equal.sum()),
            "total_adjacent_token_pairs": int(exact_equal.numel()),
            "mean_temporal_standard_deviation": float(value.std(dim=2, correction=0).mean()),
            "all_time_positions_identical": bool((value == value[:, :, :1]).all()),
        }

    centres = [PATCH_SAMPLES // 2 + PATCH_SAMPLES * index for index in range(PATCHES)]
    output = {
        "source": "32 real subject-0 trials, standardized with current training-only statistics",
        "source_kind": source_kind,
        "log_power_features_before_projection": comparison(features),
        "frequency_tokens_after_linear_17_to_192": comparison(frequency_tokens),
        "window_samples": FREQUENCY_WINDOW_SAMPLES,
        "window_centres_samples": centres,
        "window_centres_ms_within_40_440_crop": centres,
        "window_centres_ms_after_stimulus": [config["crop"][0] + value for value in centres],
        "window_step_samples_and_ms": PATCH_SAMPLES,
        "frequency_bins_hz": [float(value) for value in frequency_bin_centres()],
    }
    report = EXPERIMENT_DIR / "outputs" / "unified192" / "reports" / "frequency_token_diagnostic.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    temporary = report.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(output, indent=2), encoding="utf-8")
    os.replace(temporary, report)
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
