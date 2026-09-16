"""SVD diagnostics for the trained post-fusion Linear(192,192) layer."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch


EXPERIMENT_DIR = Path(__file__).resolve().parent
ROOT = EXPERIMENT_DIR.parents[1]
CHECKPOINT = ROOT / "checkpoints/D-095/outputs/unified192_postfusion192/runs/epoch100.pt"
REPORT = EXPERIMENT_DIR / "runs/analysis/postfusion192_svd.json"
CSV = EXPERIMENT_DIR / "runs/analysis/postfusion192_singular_values.csv"


def main() -> None:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    state = checkpoint["model"]
    key = "post_fusion_projection.weight"
    if key not in state:
        raise KeyError(f"missing {key}; available keys may indicate the wrong checkpoint")
    weight = state[key].detach().float().numpy()
    if weight.shape != (192, 192):
        raise ValueError(f"expected [192,192], got {weight.shape}")

    singular = np.linalg.svd(weight, compute_uv=False)
    squared = singular**2
    cumulative = np.cumsum(squared) / squared.sum()
    report = {
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)).replace("\\", "/"),
        "layer": key,
        "shape": list(weight.shape),
        "frobenius_norm": float(np.linalg.norm(weight, "fro")),
        "spectral_norm": float(singular[0]),
        "smallest_singular_value": float(singular[-1]),
        "condition_number": float(singular[0] / singular[-1]),
        "effective_rank_entropy": float(np.exp(-(squared / squared.sum() * np.log(np.maximum(squared / squared.sum(), 1e-30))).sum())),
        "rank_thresholds": {
            str(threshold): int(np.searchsorted(cumulative, threshold) + 1)
            for threshold in (0.90, 0.95, 0.99, 0.999)
        },
        "singular_values": [float(value) for value in singular],
        "cumulative_squared_energy": [float(value) for value in cumulative],
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    with CSV.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["index", "singular_value", "squared_energy_fraction", "cumulative_squared_energy"])
        for index, (value, fraction, total) in enumerate(zip(singular, squared / squared.sum(), cumulative), 1):
            writer.writerow([index, f"{value:.12g}", f"{fraction:.12g}", f"{total:.12g}"])
    print(json.dumps({
        "layer": key,
        "shape": list(weight.shape),
        "top_10": report["singular_values"][:10],
        "bottom_10": report["singular_values"][-10:],
        "spectral_norm": report["spectral_norm"],
        "smallest": report["smallest_singular_value"],
        "condition_number": report["condition_number"],
        "effective_rank_entropy": report["effective_rank_entropy"],
        "rank_thresholds": report["rank_thresholds"],
        "report": str(REPORT),
        "csv": str(CSV),
    }, indent=2))


if __name__ == "__main__":
    main()
