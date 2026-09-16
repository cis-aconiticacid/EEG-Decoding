"""Independent local audit of the synchronized D078 small-result snapshot."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "reports/d078_live"
REPORT = SNAPSHOT / "reports/d078_a100_continuation"
RUNS = SNAPSHOT / "runs/d078-a100-continuation"


def close(left, right, tolerance=1e-7):
    if not np.isclose(left, right, atol=tolerance, rtol=tolerance):
        raise AssertionError(f"{left} != {right}")


def metrics(rows):
    return {
        "n": len(rows),
        "latent_mse": float(np.mean([row["latent_mse"] for row in rows])),
        "class_accuracy": float(np.mean([row["class_correct"] for row in rows])),
        "image_top1": float(np.mean([row["image_correct"] for row in rows])),
        "gallery_size": 1600,
    }


def bootstrap(reference, candidate, seed):
    groups = defaultdict(list)
    for index, row in enumerate(reference):
        groups[int(row["actual_label"])].append(index)
    if len(groups) != 80 or any(len(value) != 20 for value in groups.values()):
        raise AssertionError("test category clusters differ")
    keys = sorted(groups)
    members = [np.asarray(groups[key]) for key in keys]
    changes = {
        "class_accuracy": np.asarray([r["class_correct"] for r in candidate]) - np.asarray([r["class_correct"] for r in reference]),
        "image_top1": np.asarray([r["image_correct"] for r in candidate]) - np.asarray([r["image_correct"] for r in reference]),
        "latent_mse": np.asarray([r["latent_mse"] for r in candidate]) - np.asarray([r["latent_mse"] for r in reference]),
    }
    rng = np.random.default_rng(seed)
    sampled = {key: np.empty(2000) for key in changes}
    for replicate in range(2000):
        chosen = rng.integers(0, 80, 80)
        index = np.concatenate([members[value] for value in chosen])
        for key, values in changes.items():
            sampled[key][replicate] = values[index].mean()
    return {key: {"observed_difference": float(values.mean()),
                  "percentile_95ci": np.quantile(sampled[key], [0.025, 0.975]).tolist()}
            for key, values in changes.items()}


def assert_nested(left, right, path="root"):
    if isinstance(left, dict):
        for key in left:
            assert_nested(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, list):
        if len(left) != len(right):
            raise AssertionError(f"{path}: list length")
        for index, (a, b) in enumerate(zip(left, right)):
            assert_nested(a, b, f"{path}[{index}]")
    elif isinstance(left, float):
        close(left, right)
    elif left != right:
        raise AssertionError(f"{path}: {left} != {right}")


def main():
    summary = json.loads((REPORT / "summary.json").read_text(encoding="utf-8"))
    if summary["status"] != "complete" or summary["test_evaluations_per_arm"] != 1:
        raise AssertionError("completion/test count differs")
    predictions = {"parent": json.loads((REPORT / "parent_predictions.json").read_text(encoding="utf-8"))}
    ids = [row["trial_id"] for row in predictions["parent"]]
    audited = {}
    histories = {}
    for result in summary["results"]:
        arm = result["arm"]
        if result["checkpoint_count"] != 5 or result["epoch"] != 120 or result["parent_epoch"] != 70:
            raise AssertionError(f"{arm}: endpoint/checkpoint metadata differs")
        rows = json.loads((RUNS / arm / "test_predictions.json").read_text(encoding="utf-8"))
        if [row["trial_id"] for row in rows] != ids or len({row["trial_id"] for row in rows}) != 1600:
            raise AssertionError(f"{arm}: paired test IDs differ")
        recomputed = metrics(rows)
        assert_nested(recomputed, result["test"], f"{arm}.test")
        history = json.loads((RUNS / arm / "epochs.json").read_text(encoding="utf-8"))
        if [row["epoch"] for row in history] != list(range(71, 121)):
            raise AssertionError(f"{arm}: epoch history differs")
        if any(row["coverage"] != 2400 or row["steps"] != 38 or not np.isfinite(row["loss"]) for row in history):
            raise AssertionError(f"{arm}: coverage/steps/loss differs")
        predictions[arm] = rows
        histories[arm] = history
        audited[arm] = recomputed
    for epoch_index in range(50):
        hashes = {histories[arm][epoch_index]["sampler_hash"] for arm in ("A0", "A1", "A2", "A3")}
        if len(hashes) != 1:
            raise AssertionError(f"sampler hash differs at epoch {epoch_index + 71}")
    recomputed_comparisons = {
        "A0_minus_parent": bootstrap(predictions["parent"], predictions["A0"], 78800),
        "A1_minus_A0": bootstrap(predictions["A0"], predictions["A1"], 78801),
        "A2_minus_A0": bootstrap(predictions["A0"], predictions["A2"], 78802),
        "A3_minus_A0": bootstrap(predictions["A0"], predictions["A3"], 78803),
    }
    for key, value in recomputed_comparisons.items():
        assert_nested(value, summary["comparisons"][key], key)
    if any(SNAPSHOT.rglob("*.pt")):
        raise AssertionError("small local snapshot unexpectedly contains checkpoints")
    identity = json.loads((REPORT / "identity_summary.json").read_text(encoding="utf-8"))
    if identity["status"] != "PASS" or identity["excluded_image_intersection"] != 0 or identity["sensitive_cells"] != 248:
        raise AssertionError("identity summary contract differs")
    print(json.dumps({"decision": "PASS", "metrics_recomputed": audited,
                      "bootstrap_recomputed": True, "sampler_hashes_identical": True,
                      "identity": {key: identity[key] for key in ("pairs", "fit_accuracy", "support_vectors", "finite_difference_max_abs_error")},
                      "local_checkpoint_files": 0}, indent=2))


if __name__ == "__main__":
    main()
