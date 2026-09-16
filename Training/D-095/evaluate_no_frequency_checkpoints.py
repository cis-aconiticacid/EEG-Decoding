"""Retrospective checkpoint diagnostic for the completed no-frequency run."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch


EXPERIMENT_DIR = Path(__file__).resolve().parent
ROOT = EXPERIMENT_DIR.parents[1]
sys.path.insert(0, str(ROOT))

from model import D095HybridClassifier  # noqa: E402
from run import (  # noqa: E402
    atomic_json,
    evaluate_test_once,
    load_coordinates,
    prepare_data,
    sha256,
)


def main() -> int:
    config_path = EXPERIMENT_DIR / "config/config_no_frequency.json"
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    if os.environ.get("CUDA_VISIBLE_DEVICES") != cfg["gpu_uuid"] or not torch.cuda.is_available():
        raise RuntimeError("checkpoint diagnostic requires the configured locked GPU")
    device = torch.device("cuda")
    coordinates, _ = load_coordinates(cfg)
    model = D095HybridClassifier(
        coordinates,
        classes=cfg["classes"],
        dropout=cfg["block_dropout"],
        head_dropout=cfg["head_dropout"],
        use_frequency=False,
    ).to(device)
    prepared = prepare_data(cfg, model)
    run_dir = EXPERIMENT_DIR / cfg["run_dir"]
    rows = []
    for epoch in (25, 50, 75):
        checkpoint_path = run_dir / f"epoch{epoch:03d}.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model"], strict=True)
        metrics, _ = evaluate_test_once(
            model,
            prepared["test"],
            prepared["test_labels"],
            prepared["test_indices"],
            prepared["data"],
            cfg,
            device,
            microbatch=64,
        )
        rows.append({
            "epoch": epoch,
            "checkpoint": str(checkpoint_path.relative_to(ROOT)).replace("\\", "/"),
            "checkpoint_sha256": sha256(checkpoint_path),
            "test_acc_all": metrics["acc_all"],
            "test_correct": metrics["correct"],
            "test_loss": metrics["loss"],
        })

    endpoint = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    endpoint_checkpoint = run_dir / "epoch100.pt"
    rows.append({
        "epoch": 100,
        "checkpoint": str(endpoint_checkpoint.relative_to(ROOT)).replace("\\", "/"),
        "checkpoint_sha256": sha256(endpoint_checkpoint),
        "test_acc_all": endpoint["test"]["acc_all"],
        "test_correct": endpoint["test"]["correct"],
        "test_loss": endpoint["test"]["loss"],
        "source": "original fixed-endpoint evaluation; not re-forwarded",
    })
    best = max(rows, key=lambda row: row["test_acc_all"])
    report = {
        "status": "complete",
        "purpose": "retrospective overfitting diagnosis only",
        "warning": (
            "Epochs 25, 50, and 75 were evaluated after observing the epoch-100 test result. "
            "Selecting the best row uses the official test set and is not an unbiased model-selection result."
        ),
        "new_test_forward_rounds": 3,
        "new_test_forward_samples": 3 * len(prepared["test"]),
        "rows": rows,
        "retrospective_best": best,
    }
    output_path = EXPERIMENT_DIR / cfg["report_dir"] / "checkpoint_test_diagnostic.json"
    atomic_json(output_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
