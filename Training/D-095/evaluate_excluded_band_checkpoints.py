"""Retrospective checkpoint curve for the 18--28 Hz exclusion run."""
from __future__ import annotations
import json, os, sys
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from model import D095HybridClassifier
from run import atomic_json, evaluate_test_once, load_coordinates, prepare_data, sha256


def main():
    cfg_path = HERE / "config/config_exclude18to28.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    if os.environ.get("CUDA_VISIBLE_DEVICES") != cfg["gpu_uuid"] or not torch.cuda.is_available():
        raise RuntimeError("checkpoint diagnostic requires configured GPU lock")
    device = torch.device("cuda")
    coordinates, _ = load_coordinates(cfg)
    model = D095HybridClassifier(coordinates, classes=cfg["classes"], dropout=cfg["block_dropout"],
        head_dropout=cfg["head_dropout"], use_frequency=cfg["frequency_branch"],
        excluded_band_hz=tuple(cfg["excluded_band_hz"])).to(device)
    prepared = prepare_data(cfg, model)
    run_dir = HERE / cfg["run_dir"]
    rows = []
    for epoch in (25, 50, 75):
        path = run_dir / f"epoch{epoch:03d}.pt"
        state = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"], strict=True)
        metrics, _ = evaluate_test_once(model, prepared["test"], prepared["test_labels"],
            prepared["test_indices"], prepared["data"], cfg, device, microbatch=64)
        rows.append({"epoch": epoch, "checkpoint_sha256": sha256(path),
                     "test_acc_all": metrics["acc_all"], "test_correct": metrics["correct"],
                     "test_loss": metrics["loss"]})
        print(json.dumps(rows[-1]), flush=True)
    endpoint = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    rows.append({"epoch": 100, "checkpoint_sha256": sha256(run_dir / "epoch100.pt"),
        "test_acc_all": endpoint["test"]["acc_all"], "test_correct": endpoint["test"]["correct"],
        "test_loss": endpoint["test"]["loss"], "source": "original fixed-endpoint evaluation"})
    report = {"status": "complete", "purpose": "retrospective checkpoint overfitting diagnosis",
        "warning": "Intermediate official-test checkpoints evaluated after the epoch-100 test result; do not use to select an unbiased checkpoint.",
        "new_test_forward_rounds": 3, "new_test_forward_samples": 3*len(prepared["test"]),
        "rows": rows}
    path = HERE / cfg["report_dir"] / "checkpoint_test_diagnostic.json"
    atomic_json(path, report)
    print(json.dumps(report, indent=2))

if __name__ == "__main__": main()
