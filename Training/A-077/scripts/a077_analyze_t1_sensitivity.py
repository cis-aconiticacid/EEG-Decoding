"""Read-only held-out T1 electrode permutation and saved SVM audit.

No estimator is fit or updated. Frequency results are reused from the original
held-out permutation run; this script adds grouped channel permutations only.
"""

from __future__ import annotations

import os

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "4"

import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/a077_pair_tasks"
FEATURES_PATH = ROOT / "artifacts/a077_pair_features/features.npy"
MODEL_DIR = ROOT / "artifacts/a077_pair_features/models/T1"
OUT_DIR = REPORT / "T1_support_diagnostic"
N_CHANNELS = 62
N_FREQS = 32
SEED = 77017
REPEATS = 3


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rbf_to_support(x: np.ndarray, support_x: np.ndarray, gamma: float,
                   dual: np.ndarray, intercept: float, block_rows: int = 128) -> np.ndarray:
    x64 = np.asarray(x, dtype=np.float64)
    sv64 = np.asarray(support_x, dtype=np.float64)
    sv_norm = np.einsum("ij,ij->i", sv64, sv64)
    scores = np.empty(len(x64), dtype=np.float64)
    for start in range(0, len(x64), block_rows):
        stop = min(start + block_rows, len(x64))
        left = x64[start:stop]
        distance = np.einsum("ij,ij->i", left, left)[:, None] + sv_norm[None, :] - 2.0 * (left @ sv64.T)
        np.maximum(distance, 0.0, out=distance)
        np.exp(-gamma * distance, out=distance)
        scores[start:stop] = distance @ dual + intercept
    return scores


def metric(y: np.ndarray, score: np.ndarray, threshold: float) -> tuple[float, float]:
    pred = (score >= threshold).astype(np.int8)
    return float(balanced_accuracy_score(y, pred)), float(roc_auc_score(y, score))


def score_rows(model_name: str, x: np.ndarray, bundle: dict[str, Any], svm_support_x: np.ndarray | None = None) -> np.ndarray:
    if model_name == "svm":
        model = bundle["model"]
        return rbf_to_support(x, svm_support_x, float(bundle["gamma"]),
                              np.asarray(model.dual_coef_[0]), float(model.intercept_[0]))
    return bundle["model"].predict_proba(x)[:, 1]


def main() -> None:
    started = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pair_rows = [row for row in read_csv(REPORT / "T1_pairs.csv") if row["split"] == "test"]
    test_index = np.asarray([int(row["a"]) for row in pair_rows], dtype=np.int64)
    test_index_b = np.asarray([int(row["b"]) for row in pair_rows], dtype=np.int64)
    y = np.asarray([int(row["target"]) for row in pair_rows], dtype=np.int8)
    pair_ids = [row["pair_id"] for row in pair_rows]
    feature_cache = np.load(FEATURES_PATH, mmap_mode="r", allow_pickle=False)
    x_raw = np.abs(feature_cache[test_index] - feature_cache[test_index_b]).reshape(len(pair_rows), -1).astype(np.float32)
    if x_raw.shape != (2000, N_CHANNELS * N_FREQS):
        raise ValueError(f"unexpected held-out T1 feature shape: {x_raw.shape}")

    channel_rows = read_csv(ROOT / "config/channel_map.csv")
    if len(channel_rows) != N_CHANNELS or [int(row["tensor_index"]) for row in channel_rows] != list(range(N_CHANNELS)):
        raise ValueError("channel map is not a verified 0..61 tensor-axis mapping")
    channel_names = [row["canonical_name"] for row in channel_rows]

    saved_predictions: dict[str, dict[str, Any]] = {}
    bundles: dict[str, dict[str, Any]] = {}
    test_x: dict[str, np.ndarray] = {}
    svm_support_x = None
    svm_model = None
    for model_name in ("svm", "random_forest"):
        bundle = joblib.load(MODEL_DIR / f"{model_name}.joblib")
        bundles[model_name] = bundle
        transformed = bundle["scaler"].transform(x_raw)
        test_x[model_name] = transformed
        if model_name == "svm":
            svm_model = bundle["model"]
            svm_support_x = np.asarray(bundle["x_trainval"])[np.asarray(svm_model.support_, dtype=np.int64)]
        output_dir = REPORT / "results/T1" / model_name
        predictions = read_csv(output_dir / "test_predictions.csv")
        if [row["pair_id"] for row in predictions] != pair_ids:
            raise AssertionError(f"{model_name}: saved predictions do not align with T1 test manifest")
        saved_predictions[model_name] = {
            "scores": np.asarray([float(row["score"]) for row in predictions], dtype=np.float64),
            "predictions": np.asarray([int(row["prediction"]) for row in predictions], dtype=np.int8),
            "result": json.loads((output_dir / "result.json").read_text(encoding="utf-8")),
        }

    # Verify the saved SVM's support-vector-only decision calculation exactly
    # reproduces its saved test scores before evaluating perturbations.
    svm_scores = score_rows("svm", test_x["svm"], bundles["svm"], svm_support_x)
    svm_score_error = float(np.max(np.abs(svm_scores - saved_predictions["svm"]["scores"])))
    if not np.allclose(svm_scores, saved_predictions["svm"]["scores"], rtol=1e-9, atol=1e-10):
        raise AssertionError(f"support-only SVM calculation differs from saved decision scores: {svm_score_error}")

    baseline: dict[str, dict[str, float]] = {}
    for model_name in ("svm", "random_forest"):
        threshold = 0.0 if model_name == "svm" else 0.5
        ba, auc = metric(y, saved_predictions[model_name]["scores"], threshold)
        recorded = saved_predictions[model_name]["result"]["test"]
        if abs(ba - float(recorded["balanced_accuracy"])) > 1e-12 or abs(auc - float(recorded["roc_auc"])) > 1e-12:
            raise AssertionError(f"{model_name}: baseline metrics differ from result.json")
        baseline[model_name] = {"balanced_accuracy": ba, "roc_auc": auc}

    details: list[dict[str, Any]] = []
    with threadpool_limits(limits=4):
        for channel in range(N_CHANNELS):
            cols = np.arange(channel * N_FREQS, (channel + 1) * N_FREQS, dtype=np.int64)
            for repeat in range(REPEATS):
                rng = np.random.default_rng(SEED + repeat * 1009 + channel)
                permutation = rng.permutation(len(y))
                for model_name in ("svm", "random_forest"):
                    perturbed = test_x[model_name].copy()
                    perturbed[:, cols] = test_x[model_name][permutation][:, cols]
                    scores = score_rows(model_name, perturbed, bundles[model_name], svm_support_x)
                    threshold = 0.0 if model_name == "svm" else 0.5
                    ba, auc = metric(y, scores, threshold)
                    details.append({
                        "model": model_name, "tensor_index": channel, "channel": channel_names[channel],
                        "repeat": repeat, "seed": SEED + repeat * 1009 + channel,
                        "balanced_accuracy": ba, "balanced_accuracy_drop": baseline[model_name]["balanced_accuracy"] - ba,
                        "roc_auc": auc, "roc_auc_drop": baseline[model_name]["roc_auc"] - auc,
                    })
            if (channel + 1) % 8 == 0 or channel == N_CHANNELS - 1:
                print(f"held-out channel permutations: {channel + 1}/{N_CHANNELS}", flush=True)

    detail_path = OUT_DIR / "electrode_permutation_repeats.csv"
    with detail_path.open("w", newline="", encoding="utf-8") as stream:
        fields = list(details[0])
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(details)

    summary_rows: list[dict[str, Any]] = []
    for model_name in ("svm", "random_forest"):
        for channel in range(N_CHANNELS):
            selected = [row for row in details if row["model"] == model_name and int(row["tensor_index"]) == channel]
            ba_drops = np.asarray([row["balanced_accuracy_drop"] for row in selected], dtype=np.float64)
            auc_drops = np.asarray([row["roc_auc_drop"] for row in selected], dtype=np.float64)
            summary_rows.append({
                "model": model_name, "tensor_index": channel, "channel": channel_names[channel],
                "repeats": REPEATS, "balanced_accuracy_drop_mean": float(ba_drops.mean()),
                "balanced_accuracy_drop_sd": float(ba_drops.std(ddof=1)),
                "roc_auc_drop_mean": float(auc_drops.mean()), "roc_auc_drop_sd": float(auc_drops.std(ddof=1)),
                "baseline_balanced_accuracy": baseline[model_name]["balanced_accuracy"],
                "baseline_roc_auc": baseline[model_name]["roc_auc"],
                "scope": "exploratory held-out grouped permutation; all 32 frequency columns for this channel shuffled together",
            })
    summary_path = OUT_DIR / "electrode_permutation.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    frequency_summary = {}
    for model_name in ("svm", "random_forest"):
        frequency_rows = read_csv(REPORT / "results/T1" / model_name / "frequency_permutation.csv")
        frequency_summary[model_name] = sorted(
            [{"frequency_hz": float(row["frequency_hz"]), "balanced_accuracy_drop": float(row["balanced_accuracy_drop"]),
              "balanced_accuracy_after_permutation": float(row["balanced_accuracy_after_permutation"])} for row in frequency_rows],
            key=lambda row: row["balanced_accuracy_drop"], reverse=True,
        )

    support_summary = json.loads((OUT_DIR / "summary.json").read_text(encoding="utf-8"))
    output = {
        "status": "complete_no_fit",
        "task": "T1 only",
        "test_pairs": len(y),
        "test_pair_manifest_sha256": sha_file(REPORT / "T1_pairs.csv"),
        "models": baseline,
        "svm_support_score_max_abs_difference_from_saved_predictions": svm_score_error,
        "electrode_permutation": {
            "repeats_per_channel": REPEATS, "seed_base": SEED,
            "channel_names": "config/channel_map.csv; tensor-axis mapping follows the official-RGNN inferred contract and is not embedded in the archive",
            "results_csv": str(summary_path.relative_to(ROOT)),
            "repeat_level_csv": str(detail_path.relative_to(ROOT)),
            "permutation_note": "held-out, post-hoc model sensitivity only; not causal electrode relevance",
        },
        "existing_held_out_frequency_permutation_top8": {key: rows[:8] for key, rows in frequency_summary.items()},
        "existing_svm_local_gradient_summary": {
            "scope": support_summary["scope"],
            "top_frequencies": support_summary["top_frequencies"],
            "top_electrodes": support_summary["top_electrodes"],
            "finite_difference_max_error": support_summary["finite_difference_max_error"],
            "interpretation": "local standardized-feature decision gradient; complementary to held-out permutation and not localization",
        },
        "elapsed_seconds": time.time() - started,
    }
    temp = OUT_DIR / "electrode_permutation_summary.json.tmp"
    temp.write_text(json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(OUT_DIR / "electrode_permutation_summary.json")
    print(json.dumps({"status": output["status"], "test_pairs": len(y), "baseline": baseline,
                      "svm_score_check_max_error": svm_score_error, "elapsed_seconds": output["elapsed_seconds"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main()
