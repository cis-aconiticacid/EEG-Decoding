"""Fit D078's leak-excluded identity auxiliary model and freeze 248 cells.

This CPU-only stage reads the two safe mmap archives, excludes every subject-0
test image ID, and performs no image-decoder training.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "4"

import joblib
import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from eegdecoding.data import PART_NAMES, ensure_trial_manifest, load_part  # noqa: E402

CFG_PATH = ROOT / "config/d078_a100_continuation.json"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def thash(array: np.ndarray | torch.Tensor) -> str:
    if isinstance(array, torch.Tensor):
        array = array.detach().cpu().contiguous().numpy()
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def status(report: Path, phase: str, **fields: Any) -> None:
    value = {"task": "D078-identity", "phase": phase, "pid": os.getpid(), "updated_unix": time.time(), **fields}
    atomic_json(report / "identity_status.json", value)
    print(json.dumps(value, ensure_ascii=False, allow_nan=False), flush=True)


def frequency(raw: torch.Tensor) -> torch.Tensor:
    window = torch.hann_window(400, periodic=True, dtype=raw.dtype, device=raw.device)
    fft = torch.fft.rfft((raw - raw.mean(-1, keepdim=True)) * window)
    return (2 * fft[..., 1:33].abs().square() / (1000 * window.square().sum())).clamp_min(1e-30).log10()


def load_rows(cfg: dict[str, Any]) -> tuple[list[dict[str, str]], list[dict[str, Any]], set[str], set[str]]:
    with (ROOT / cfg["manifest"]).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    order = {name: index for index, name in enumerate(PART_NAMES)}
    rows.sort(key=lambda row: (order[row["source_file"]], int(row["source_index"])))
    if len(rows) != 63850 or {int(row["subject"]) for row in rows} != set(range(16)):
        raise AssertionError("expected 63,850 trials and subjects 0..15")
    s0 = [row for row in rows if int(row["subject"]) == 0]
    train_images = {row["image"] for row in s0 if int(row["position_in_class_block"]) < 30}
    excluded_images = {row["image"] for row in s0 if int(row["position_in_class_block"]) >= 30}
    if len(train_images) != 2400 or len(excluded_images) != 1600 or train_images & excluded_images:
        raise AssertionError("subject0 first30/last20 image split differs")
    eligible: list[dict[str, Any]] = []
    for row in rows:
        if row["image"] in train_images:
            copy = dict(row)
            copy["feature_row"] = len(eligible)
            eligible.append(copy)
    if any(row["image"] in excluded_images for row in eligible):
        raise AssertionError("excluded subject0 test image entered identity pool")
    if {row["image"] for row in eligible} != train_images:
        raise AssertionError("identity pool does not cover exactly the 2400 training image IDs")
    return rows, eligible, train_images, excluded_images


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fields = fields or list(rows[0])
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)


def extract_features(cfg: dict[str, Any], eligible: list[dict[str, Any]], artifact: Path, report: Path) -> np.ndarray:
    path = artifact / "eligible_features.npy"
    meta_path = artifact / "eligible_features.json"
    manifest_hash = sha(ROOT / cfg["manifest"])
    if path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if (meta.get("manifest_sha256") != manifest_hash or meta.get("extractor_sha256") != sha(Path(__file__))
                or list(array.shape) != [len(eligible), 62, 32] or array.dtype != np.float32):
            del array
            path.unlink()
            meta_path.unlink()
        else:
            return array

    artifact.mkdir(parents=True, exist_ok=True)
    partial = artifact / "eligible_features.partial.npy"
    if partial.exists():
        partial.unlink()
    output = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float32, shape=(len(eligible), 62, 32))
    by_file: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        by_file[row["source_file"]].append(row)
    torch.set_num_threads(4)
    for filename in PART_NAMES:
        part_rows = sorted(by_file[filename], key=lambda row: int(row["source_index"]))
        archive = load_part(ROOT / filename)
        status(report, "extracting_identity_features", source_file=filename, rows=len(part_rows))
        for start in range(0, len(part_rows), 64):
            batch_rows = part_rows[start : start + 64]
            raw = torch.stack([
                archive["dataset"][int(row["source_index"])]["eeg_data"][:, 40:440].float()
                for row in batch_rows
            ])
            transformed = frequency(raw).numpy()
            output[[int(row["feature_row"]) for row in batch_rows]] = transformed
        output.flush()
        del archive
    del output
    os.replace(partial, path)
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if not np.isfinite(array).all():
        raise AssertionError("identity features contain nonfinite values")
    atomic_json(meta_path, {
        "shape": list(array.shape), "dtype": str(array.dtype), "manifest_sha256": manifest_hash,
        "extractor_sha256": sha(Path(__file__)),
        "eligible_trial_rows": len(eligible), "eligible_image_ids": 2400,
        "excluded_subject0_test_image_ids": 1600, "feature_sha256": sha(path),
        "transform": "40:440; demean; periodic Hann; one-sided log10 PSD 2.5:2.5:80 Hz",
    })
    return array


def make_pairs(eligible: list[dict[str, Any]], seed: int, report: Path) -> list[dict[str, Any]]:
    by_label_image: dict[tuple[int, str], dict[int, dict[str, Any]]] = defaultdict(dict)
    images_by_label: dict[int, set[str]] = defaultdict(set)
    for row in eligible:
        key = (int(row["label_index"]), str(row["image"]))
        subject = int(row["subject"])
        if subject in by_label_image[key]:
            raise AssertionError("duplicate subject/category/image trial")
        by_label_image[key][subject] = row
        images_by_label[key[0]].add(key[1])
    if set(images_by_label) != set(range(80)) or any(len(images) != 30 for images in images_by_label.values()):
        raise AssertionError("identity pool must contain 30 images for each of 80 categories")

    rng = np.random.default_rng(seed)
    positive_usage: Counter[int] = Counter()
    negative_usage: Counter[int] = Counter()
    pairs: list[dict[str, Any]] = []
    for label in range(80):
        candidates = list(itertools.combinations(sorted(images_by_label[label]), 2))
        rng.shuffle(candidates)
        selected = candidates[:50]
        if len(selected) != 50:
            raise AssertionError("insufficient identity matched image pairs")
        for within_label, (image_a, image_b) in enumerate(selected):
            rows_a, rows_b = by_label_image[(label, image_a)], by_label_image[(label, image_b)]
            common = sorted(set(rows_a) & set(rows_b))
            minimum = min(positive_usage[subject] for subject in common)
            choices = [subject for subject in common if positive_usage[subject] == minimum]
            positive_subject = int(rng.choice(choices))
            positive_usage[positive_subject] += 2
            cross = [(a, b) for a in rows_a for b in rows_b if a != b]
            minimum = min(negative_usage[a] + negative_usage[b] for a, b in cross)
            choices2 = [(a, b) for a, b in cross if negative_usage[a] + negative_usage[b] == minimum]
            negative_a, negative_b = choices2[int(rng.integers(len(choices2)))]
            negative_usage[negative_a] += 1
            negative_usage[negative_b] += 1
            match_id = f"D078-ID-{label:02d}-{within_label:02d}"
            for target, left, right in (
                (1, rows_a[positive_subject], rows_b[positive_subject]),
                (0, rows_a[negative_a], rows_b[negative_b]),
            ):
                pairs.append({
                    "pair_id": f"D078-ID-{len(pairs):05d}", "match_id": match_id, "target": target,
                    "a": int(left["feature_row"]), "b": int(right["feature_row"]),
                    "trial_a": left["trial_id"], "trial_b": right["trial_id"],
                    "image_a": image_a, "image_b": image_b, "label_index": label,
                    "subject_a": int(left["subject"]), "subject_b": int(right["subject"]),
                    "source_file_a": left["source_file"], "source_index_a": int(left["source_index"]),
                    "source_file_b": right["source_file"], "source_index_b": int(right["source_index"]),
                })
    if len(pairs) != 8000 or sum(int(row["target"]) for row in pairs) != 4000:
        raise AssertionError("identity pair count/balance differs")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pairs:
        groups[row["match_id"]].append(row)
        if bool(int(row["target"])) != (int(row["subject_a"]) == int(row["subject_b"])):
            raise AssertionError("identity label mismatch")
    for rows in groups.values():
        if len(rows) != 2 or {int(row["target"]) for row in rows} != {0, 1}:
            raise AssertionError("matched identity group mismatch")
        if {(row["image_a"], row["image_b"]) for row in rows} != {(rows[0]["image_a"], rows[0]["image_b"])}:
            raise AssertionError("positive/negative image pair mismatch")
    write_csv(report / "identity_pairs.csv", pairs)
    return pairs


def kernel(a: np.ndarray, b: np.ndarray, gamma: float, block_rows: int = 256) -> np.ndarray:
    a64, b64 = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    norm_b = np.einsum("ij,ij->i", b64, b64)
    output = np.empty((len(a64), len(b64)), dtype=np.float64)
    for start in range(0, len(a64), block_rows):
        stop = min(start + block_rows, len(a64))
        left = a64[start:stop]
        distance = np.einsum("ij,ij->i", left, left)[:, None] + norm_b[None] - 2.0 * (left @ b64.T)
        np.maximum(distance, 0.0, out=distance)
        output[start:stop] = np.exp(-gamma * distance)
    return output


def score_support(x: np.ndarray, support: np.ndarray, dual: np.ndarray, intercept: float, gamma: float) -> np.ndarray:
    return kernel(x, support, gamma, block_rows=64) @ dual + intercept


def gradient(x: np.ndarray, support: np.ndarray, dual: np.ndarray, gamma: float) -> np.ndarray:
    x64, support64 = np.asarray(x, dtype=np.float64), np.asarray(support, dtype=np.float64)
    output = np.empty_like(x64)
    for start in range(0, len(x64), 32):
        stop = min(start + 32, len(x64))
        left = x64[start:stop]
        k = kernel(left, support64, gamma, block_rows=32)
        weighted = k * dual[None]
        output[start:stop] = 2.0 * gamma * (weighted @ support64 - weighted.sum(1, keepdims=True) * left)
    return output


def fit_identity(cfg: dict[str, Any], features: np.ndarray, pairs: list[dict[str, Any]], artifact: Path, report: Path) -> dict[str, Any]:
    a = np.asarray([int(row["a"]) for row in pairs], dtype=np.int64)
    b = np.asarray([int(row["b"]) for row in pairs], dtype=np.int64)
    y = np.asarray([int(row["target"]) for row in pairs], dtype=np.int8)
    raw = np.abs(features[a] - features[b]).reshape(len(pairs), -1).astype(np.float32, copy=False)
    scaler = StandardScaler().fit(raw)
    x = scaler.transform(raw).astype(np.float32, copy=False)
    gamma = float(cfg["identity_fit"]["gamma"])
    c_value = float(cfg["identity_fit"]["C"])
    status(report, "fitting_identity_svm", pairs=len(y), dimension=x.shape[1])
    with threadpool_limits(limits=4):
        gram = kernel(x, x, gamma)
        model = SVC(C=c_value, kernel="precomputed", cache_size=512, tol=1e-3).fit(gram, y)
        fit_score = model.decision_function(gram)
    fit_pred = (fit_score >= 0).astype(np.int8)

    parity_train = np.arange(512)
    parity_eval = np.arange(512, 640)
    with threadpool_limits(limits=4):
        small_gram = kernel(x[parity_train], x[parity_train], gamma)
        precomputed = SVC(C=c_value, kernel="precomputed", tol=1e-3).fit(small_gram, y[parity_train])
        precomputed_score = precomputed.decision_function(kernel(x[parity_eval], x[parity_train], gamma))
        direct = SVC(C=c_value, kernel="rbf", gamma=gamma, tol=1e-3).fit(x[parity_train], y[parity_train])
        direct_score = direct.decision_function(x[parity_eval])
    parity_error = float(np.max(np.abs(precomputed_score - direct_score)))
    if parity_error > 1e-9 or not np.array_equal(precomputed_score >= 0, direct_score >= 0):
        raise AssertionError(f"precomputed/direct RBF parity failed: {parity_error}")

    rng = np.random.default_rng(int(cfg["identity_seed"]) + 1)
    subset = np.r_[rng.choice(np.flatnonzero(y == 0), 256, replace=False),
                   rng.choice(np.flatnonzero(y == 1), 256, replace=False)]
    rng.shuffle(subset)
    support = x[model.support_]
    dual = np.asarray(model.dual_coef_[0], dtype=np.float64)
    analytic = gradient(x[subset], support, dual, gamma)
    sensitivity = np.square(analytic).mean(0).reshape(62, 32)
    count = int(cfg["identity_fit"]["sensitive_cells"])
    ranked = np.argsort(-sensitivity.reshape(-1), kind="stable")
    selected = np.sort(ranked[:count]).astype(np.int64)
    mask = np.zeros(1984, dtype=np.bool_)
    mask[selected] = True

    checks = []
    eps = 1e-4
    check_rows = subset[:4]
    check_features = ranked[:4]
    for row_index, feature_index in zip(check_rows, check_features):
        plus = x[row_index : row_index + 1].astype(np.float64, copy=True)
        minus = x[row_index : row_index + 1].astype(np.float64, copy=True)
        plus[0, feature_index] += eps
        minus[0, feature_index] -= eps
        numeric = (score_support(plus, support, dual, float(model.intercept_[0]), gamma)[0]
                   - score_support(minus, support, dual, float(model.intercept_[0]), gamma)[0]) / (2 * eps)
        actual = analytic[np.flatnonzero(subset == row_index)[0], feature_index]
        checks.append(abs(float(actual - numeric)))
    max_fd_error = max(checks)
    if max_fd_error > 1e-6:
        raise AssertionError(f"identity gradient finite-difference mismatch: {max_fd_error}")

    np.save(artifact / "sensitive_mask.npy", mask.reshape(62, 32), allow_pickle=False)
    np.save(artifact / "identity_scaler_mean.npy", scaler.mean_, allow_pickle=False)
    np.save(artifact / "identity_scaler_scale.npy", scaler.scale_, allow_pickle=False)
    model_path = artifact / "identity_svm.joblib"
    joblib.dump({"model": model, "x_train": x, "pair_ids": [row["pair_id"] for row in pairs],
                 "gamma": gamma, "C": c_value}, model_path, compress=0)
    cells = [{"flat_index": int(index), "electrode_index": int(index // 32),
              "frequency_index": int(index % 32), "frequency_hz": float(2.5 * (index % 32 + 1)),
              "mean_squared_gradient": float(sensitivity.reshape(-1)[index])} for index in ranked[:count]]
    atomic_json(report / "sensitive_cells.json", {"count": count, "cells_ranked": cells})
    summary = {
        "status": "PASS", "config_sha256": sha(CFG_PATH), "code_sha256": sha(Path(__file__)),
        "eligible_trial_rows": int(len(features)), "eligible_image_ids": 2400,
        "excluded_subject0_test_image_ids": 1600, "excluded_image_intersection": 0,
        "pairs": len(pairs), "positive_pairs": int(y.sum()), "negative_pairs": int((y == 0).sum()),
        "categories": 80, "subjects": sorted({int(row[k]) for row in pairs for k in ("subject_a", "subject_b")}),
        "fit_accuracy": float(accuracy_score(y, fit_pred)), "support_vectors": int(len(model.support_)),
        "C": c_value, "gamma": gamma, "scaler_fit_pairs": len(y), "probe_pairs": len(subset),
        "precomputed_direct_rbf_max_abs_error": parity_error,
        "sensitive_cells": count, "finite_difference_max_abs_error": max_fd_error,
        "pair_manifest_sha256": sha(report / "identity_pairs.csv"),
        "feature_cache_sha256": sha(artifact / "eligible_features.npy"),
        "sensitive_mask_sha256": sha(artifact / "sensitive_mask.npy"),
        "model_sha256": sha(model_path), "feature_map_sha256": thash(sensitivity),
        "identity_pool_policy": "all subjects restricted to subject0 first30 image IDs; subject0 last20 IDs excluded",
    }
    atomic_json(report / "identity_summary.json", summary)
    atomic_json(artifact / "completed.json", summary)
    del gram
    return summary


def self_test() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(12, 8))
    support = rng.normal(size=(7, 8))
    dual = rng.normal(size=7)
    gamma = 0.125
    analytic = gradient(x[:2], support, dual, gamma)
    eps = 1e-5
    plus, minus = x[:1].copy(), x[:1].copy()
    plus[0, 3] += eps
    minus[0, 3] -= eps
    numeric = (score_support(plus, support, dual, 0.3, gamma)[0] - score_support(minus, support, dual, 0.3, gamma)[0]) / (2 * eps)
    if abs(numeric - analytic[0, 3]) > 1e-8:
        raise AssertionError("synthetic gradient check failed")
    print(json.dumps({"self_test": "PASS", "gradient_error": abs(float(numeric - analytic[0, 3]))}), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit", action="store_true")
    args = parser.parse_args()
    self_test()
    if not args.fit:
        return 0
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, ""):
        raise RuntimeError("identity auxiliary fit must run CPU-only")
    cfg = json.loads(CFG_PATH.read_text(encoding="utf-8"))
    ensure_trial_manifest()
    report = ROOT / cfg["report_dir"]
    artifact = ROOT / cfg["identity_artifact_dir"]
    if (artifact / "completed.json").exists():
        completed = json.loads((artifact / "completed.json").read_text(encoding="utf-8"))
        if completed.get("config_sha256") != sha(CFG_PATH) or completed.get("code_sha256") != sha(Path(__file__)):
            raise RuntimeError("stale completed identity fit")
        print(json.dumps(completed, indent=2), flush=True)
        return 0
    _, eligible, train_images, excluded_images = load_rows(cfg)
    write_csv(report / "identity_eligible_trials.csv", eligible)
    atomic_json(report / "identity_image_split.json", {
        "train_image_ids": sorted(train_images), "excluded_test_image_ids": sorted(excluded_images),
        "intersection": sorted(train_images & excluded_images),
    })
    status(report, "identity_pool_ready", eligible_trials=len(eligible), train_images=len(train_images), excluded_images=len(excluded_images))
    features = extract_features(cfg, eligible, artifact, report)
    pairs = make_pairs(eligible, int(cfg["identity_seed"]), report)
    summary = fit_identity(cfg, features, pairs, artifact, report)
    status(report, "completed", **summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
