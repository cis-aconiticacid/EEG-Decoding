"""Run audited A077 T1/T3 SVM and random-forest pair classifiers.

The runner refuses to fit anything until Sol records an explicit PASS in
reports/a077_pair_tasks/PRE_RUN_REVIEW.md. T2 is intentionally never fitted.
"""

from __future__ import annotations

import os

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "4"

import csv
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config/a077_pair_tasks.json"
REPORT_DIR = ROOT / "reports/a077_pair_tasks"
FEATURE_DIR = ROOT / "artifacts/a077_pair_features"
MODEL_DIR = FEATURE_DIR / "models"
RESULT_DIR = REPORT_DIR / "results"
T2_TASK_NAME = "T2_same_person_same_true_session"


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def _read_pairs(task: str) -> tuple[list[dict[str, str]], dict[str, str]]:
    path = REPORT_DIR / f"{task}_pairs.csv"
    if not path.exists():
        raise FileNotFoundError(f"required pair manifest missing: {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    splits = {row["split"] for row in rows}
    if splits != {"train", "validation", "test"}:
        raise ValueError(f"{task}: pair manifest split mismatch: {splits}")
    for split in splits:
        labels = [int(row["target"]) for row in rows if row["split"] == split]
        if labels.count(0) != labels.count(1) or not labels:
            raise ValueError(f"{task}/{split}: expected balanced nonempty pair data")
    return rows, {"manifest": str(path.relative_to(ROOT)), "sha256": _sha_file(path)}


def _review_gate() -> None:
    ready = REPORT_DIR / "PRE_RUN_READY.md"
    review = REPORT_DIR / "PRE_RUN_REVIEW.md"
    if not ready.exists():
        raise RuntimeError("PRE_RUN_READY.md is missing")
    if not review.exists():
        raise RuntimeError("model fitting blocked: waiting for Sol's PRE_RUN_REVIEW.md")
    content = review.read_text(encoding="utf-8")
    if not any(line.strip().lower() == "decision: pass" for line in content.splitlines()):
        raise RuntimeError("model fitting blocked: PRE_RUN_REVIEW.md must contain the explicit line 'Decision: PASS'")


def _pair_arrays(rows: list[dict[str, str]], features: np.ndarray) -> dict[str, dict[str, Any]]:
    result = {}
    for split in ("train", "validation", "test"):
        subset = [row for row in rows if row["split"] == split]
        a = np.fromiter((int(row["a"]) for row in subset), dtype=np.int64, count=len(subset))
        b = np.fromiter((int(row["b"]) for row in subset), dtype=np.int64, count=len(subset))
        x = np.abs(features[a] - features[b]).reshape(len(subset), -1).astype(np.float32, copy=False)
        if not np.isfinite(x).all():
            raise ValueError(f"nonfinite pair feature in {split}")
        result[split] = {
            "rows": subset,
            "x": x,
            "y": np.fromiter((int(row["target"]) for row in subset), dtype=np.int8, count=len(subset)),
        }
    return result


def _metric_summary(y: np.ndarray, pred: np.ndarray, score: np.ndarray) -> dict[str, Any]:
    cm = confusion_matrix(y, pred, labels=[0, 1])
    return {
        "n": int(len(y)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "roc_auc": float(roc_auc_score(y, score)),
        "negative_recall_specificity": float(recall_score(y, pred, pos_label=0, zero_division=0)),
        "positive_recall_sensitivity": float(recall_score(y, pred, pos_label=1, zero_division=0)),
        "confusion_matrix_rows_true_0_1_cols_pred_0_1": cm.tolist(),
    }


def _group_metrics(task: str, pair_rows: list[dict[str, str]], y: np.ndarray, pred: np.ndarray, score: np.ndarray) -> dict[str, Any]:
    by_category: dict[str, list[int]] = defaultdict(list)
    by_subject: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(pair_rows):
        by_category[str(row["category"])].append(i)
        if task == "T1":
            for subject in {str(row["subject_a"]), str(row["subject_b"])}:
                by_subject[subject].append(i)
        else:
            by_subject[str(row["subject_a"])].append(i)

    def summarize(groups: dict[str, list[int]]) -> dict[str, Any]:
        values = {}
        for group, indices in sorted(groups.items(), key=lambda item: item[0]):
            ix = np.asarray(indices, dtype=np.int64)
            if np.unique(y[ix]).size == 2:
                values[group] = _metric_summary(y[ix], pred[ix], score[ix])
        return values

    return {"by_category": summarize(by_category), "by_subject": summarize(by_subject)}


def _cluster_bootstrap(pair_rows: list[dict[str, str]], y: np.ndarray, pred: np.ndarray, score: np.ndarray,
                       seed: int, replicates: int = 2000) -> dict[str, Any]:
    clusters: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(pair_rows):
        clusters[str(row["match_id"])].append(i)
    keys = sorted(clusters)
    if len(keys) < 2 or any(len(clusters[key]) != 2 for key in keys):
        raise ValueError("expected independent matched two-pair clusters for bootstrap")
    rng = np.random.default_rng(seed)
    ba = np.empty(replicates, dtype=np.float64)
    auc = np.empty(replicates, dtype=np.float64)
    members = [np.asarray(clusters[key], dtype=np.int64) for key in keys]
    for b in range(replicates):
        chosen = rng.integers(0, len(members), size=len(members))
        ix = np.concatenate([members[j] for j in chosen])
        ba[b] = balanced_accuracy_score(y[ix], pred[ix])
        auc[b] = roc_auc_score(y[ix], score[ix])
    return {
        "cluster_unit": "matched image-pair group for T1; matched subject/category/center group for T3",
        "clusters": len(keys), "replicates": replicates, "seed": seed,
        "balanced_accuracy_percentile_95ci": np.quantile(ba, [0.025, 0.975]).tolist(),
        "roc_auc_percentile_95ci": np.quantile(auc, [0.025, 0.975]).tolist(),
    }


def _save_predictions(path: Path, rows: list[dict[str, str]], pred: np.ndarray, score: np.ndarray) -> None:
    fields = ["pair_id", "match_id", "split", "target", "prediction", "score", "target_name", "category_index", "category",
              "image_a", "image_b", "subject_a", "subject_b", "position_a", "position_b", "center", "gap"]
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row, p, s in zip(rows, pred, score):
            out = {key: row[key] for key in fields if key in row}
            out["prediction"] = int(p)
            out["score"] = float(s)
            writer.writerow(out)
    temp.replace(path)


def _frequency_permutation(task: str, model_name: str, model: Any, x_test: np.ndarray, y: np.ndarray,
                           score_fn: Any, seed: int) -> list[dict[str, Any]]:
    base_score = np.asarray(score_fn(x_test), dtype=np.float64)
    base_pred = (base_score >= 0.5).astype(np.int8) if model_name == "random_forest" else (base_score >= 0.0).astype(np.int8)
    base = balanced_accuracy_score(y, base_pred)
    rng = np.random.default_rng(seed)
    output = []
    for freq in range(32):
        permuted = x_test.copy()
        indices = np.arange(len(x_test))
        rng.shuffle(indices)
        cols = np.arange(freq, x_test.shape[1], 32)
        permuted[:, cols] = x_test[indices][:, cols]
        score = np.asarray(score_fn(permuted), dtype=np.float64)
        pred = (score >= 0.5).astype(np.int8) if model_name == "random_forest" else (score >= 0.0).astype(np.int8)
        output.append({
            "frequency_hz": float(2.5 * (freq + 1)), "balanced_accuracy_after_permutation": float(balanced_accuracy_score(y, pred)),
            "balanced_accuracy_drop": float(base - balanced_accuracy_score(y, pred)),
            "note": "exploratory held-out grouped permutation; not a causal or unbiased frequency-importance estimate",
        })
    return output


def _kernel(a: np.ndarray, b: np.ndarray, gamma: float, block_rows: int = 256) -> np.ndarray:
    a64 = np.asarray(a, dtype=np.float64)
    b64 = np.asarray(b, dtype=np.float64)
    norm_b = np.einsum("ij,ij->i", b64, b64)
    output = np.empty((len(a64), len(b64)), dtype=np.float64)
    for start in range(0, len(a64), block_rows):
        stop = min(start + block_rows, len(a64))
        left = a64[start:stop]
        dist = np.einsum("ij,ij->i", left, left)[:, None] + norm_b[None, :] - 2.0 * (left @ b64.T)
        np.maximum(dist, 0.0, out=dist)
        dist *= -gamma
        np.exp(dist, out=dist)
        output[start:stop] = dist
    return output


def _decision_parity_check(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, c_value: float, gamma: float) -> dict[str, Any]:
    n_train = min(512, len(x_train))
    n_val = min(128, len(x_val))
    xs, ys, xv = x_train[:n_train], y_train[:n_train], x_val[:n_val]
    gram = _kernel(xs, xs, gamma)
    val_gram = _kernel(xv, xs, gamma)
    precomputed = SVC(C=c_value, kernel="precomputed", cache_size=256, tol=1e-7).fit(gram, ys)
    direct = SVC(C=c_value, kernel="rbf", gamma=gamma, cache_size=256, tol=1e-7).fit(xs, ys)
    pre_scores = precomputed.decision_function(val_gram)
    direct_scores = direct.decision_function(xv)
    error = float(np.max(np.abs(pre_scores - direct_scores)))
    passed = bool(np.allclose(pre_scores, direct_scores, rtol=1e-8, atol=1e-9))
    if not passed:
        raise AssertionError(f"precomputed RBF decision parity failed: max_abs_error={error}")
    return {"passed": passed, "train_rows": n_train, "validation_rows": n_val, "max_abs_decision_error": error,
            "comparison": "sklearn SVC(kernel='precomputed') vs SVC(kernel='rbf') on identical deterministic subset"}


def _selection_key(value: dict[str, Any], complexity: float) -> tuple[float, float, float]:
    return (float(value["balanced_accuracy"]), float(value["roc_auc"]), -complexity)


def _fit_svm(task: str, arrays: dict[str, dict[str, Any]], signature: str, base_seed: int) -> None:
    out = RESULT_DIR / task / "svm"
    model_path = MODEL_DIR / task / "svm.joblib"
    out.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    state_path = out / "candidate_results.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("signature") != signature:
            raise RuntimeError(f"stale SVM candidate results in {out}; inspect and move before rerunning")
    else:
        state = {"signature": signature, "candidates": {}}

    x_train_raw, y_train = arrays["train"]["x"], arrays["train"]["y"]
    x_val_raw, y_val = arrays["validation"]["x"], arrays["validation"]["y"]
    scaler = StandardScaler().fit(x_train_raw)
    x_train = scaler.transform(x_train_raw)
    x_val = scaler.transform(x_val_raw)
    gamma = 1.0 / x_train.shape[1]
    with threadpool_limits(limits=4):
        k_train = _kernel(x_train, x_train, gamma)
        k_val = _kernel(x_val, x_train, gamma)
        for c_value in (0.1, 1.0, 10.0):
            key = str(c_value)
            pred_path = out / f"C_{key}_validation.npz"
            if key in state["candidates"] and pred_path.exists():
                continue
            print(f"{task} SVM validation C={c_value:g}: fitting", flush=True)
            clf = SVC(C=c_value, kernel="precomputed", cache_size=512, tol=1e-3)
            clf.fit(k_train, y_train)
            scores = clf.decision_function(k_val)
            pred = (scores >= 0.0).astype(np.int8)
            metrics = _metric_summary(y_val, pred, scores)
            np.savez_compressed(pred_path, prediction=pred, score=scores, labels=y_val)
            state["candidates"][key] = metrics
            _write_json(state_path, state)
            print(f"{task} SVM C={c_value:g} validation balanced_accuracy={metrics['balanced_accuracy']:.6f} auc={metrics['roc_auc']:.6f}", flush=True)
        del k_train, k_val

    best_c = max((0.1, 1.0, 10.0), key=lambda c: _selection_key(state["candidates"][str(c)], c))
    parity = _decision_parity_check(x_train, y_train, x_val, best_c, gamma)
    trainval_raw = np.concatenate((arrays["train"]["x"], arrays["validation"]["x"]), axis=0)
    trainval_y = np.concatenate((y_train, y_val))
    trainval_rows = arrays["train"]["rows"] + arrays["validation"]["rows"]
    if model_path.exists():
        bundle = joblib.load(model_path)
        if bundle.get("signature") != signature:
            raise RuntimeError(f"stale final SVM model in {model_path}")
        clf, final_scaler, x_trainval = bundle["model"], bundle["scaler"], bundle["x_trainval"]
    else:
        final_scaler = StandardScaler().fit(trainval_raw)
        x_trainval = final_scaler.transform(trainval_raw)
        print(f"{task} SVM final refit on train+validation; C={best_c:g}", flush=True)
        with threadpool_limits(limits=4):
            k_trainval = _kernel(x_trainval, x_trainval, gamma)
            clf = SVC(C=best_c, kernel="precomputed", cache_size=512, tol=1e-3).fit(k_trainval, trainval_y)
        bundle = {"signature": signature, "model": clf, "scaler": final_scaler, "x_trainval": x_trainval,
                  "gamma": gamma, "best_C": best_c, "trainval_pair_ids": [r["pair_id"] for r in trainval_rows]}
        temp_model = model_path.with_suffix(".joblib.tmp")
        joblib.dump(bundle, temp_model, compress=0)
        temp_model.replace(model_path)
        del k_trainval

    x_test = final_scaler.transform(arrays["test"]["x"])
    with threadpool_limits(limits=4):
        test_gram = _kernel(x_test, x_trainval, gamma)
        test_scores = clf.decision_function(test_gram)
        test_pred = (test_scores >= 0.0).astype(np.int8)
    y_test = arrays["test"]["y"]
    test_rows = arrays["test"]["rows"]
    metrics = _metric_summary(y_test, test_pred, test_scores)
    result = {
        "task": task, "model": "RBF SVM", "signature": signature, "selected_C": best_c,
        "gamma": gamma, "feature_dimension": int(x_train.shape[1]),
        "selection_metric": "validation balanced accuracy; ROC AUC tie-break", "validation_candidates": state["candidates"],
        "rbf_decision_function_parity": parity,
        "train_pairs": len(arrays["train"]["y"]), "validation_pairs": len(y_val),
        "final_fit_pairs_train_plus_validation": len(trainval_y), "test_pairs": len(y_test),
        "support_vectors": int(len(clf.support_)), "support_fraction": float(len(clf.support_) / len(trainval_y)),
        "test": metrics,
        "grouped_test_metrics": _group_metrics(task, test_rows, y_test, test_pred, test_scores),
        "cluster_bootstrap": _cluster_bootstrap(test_rows, y_test, test_pred, test_scores, base_seed + 301),
        "frequency_permutation": "see frequency_permutation.csv; exploratory only",
        "support_pair_mapping": "support_vectors.csv maps precomputed-kernel support indices to train+validation pair IDs",
        "model_file": str(model_path.relative_to(ROOT)),
    }
    pred_path = out / "test_predictions.csv"
    _save_predictions(pred_path, test_rows, test_pred, test_scores)
    support_path = out / "support_vectors.csv"
    with support_path.open("w", newline="", encoding="utf-8") as stream:
        fields = list(trainval_rows[0].keys()) + ["support_index", "dual_coefficient"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for support_index, dual in zip(clf.support_, clf.dual_coef_[0]):
            row = dict(trainval_rows[int(support_index)])
            row["support_index"] = int(support_index)
            row["dual_coefficient"] = float(dual)
            writer.writerow(row)

    def score_fn(values: np.ndarray) -> np.ndarray:
        gram = _kernel(values, x_trainval, gamma)
        return clf.decision_function(gram)

    permutation = _frequency_permutation(task, "svm", clf, x_test, y_test, score_fn, base_seed + 401)
    with (out / "frequency_permutation.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(permutation[0]))
        writer.writeheader()
        writer.writerows(permutation)
    result["frequency_permutation_scope"] = "32 grouped 2.5-Hz feature-column permutations on held-out pairs; exploratory post-hoc diagnostic"
    _write_json(out / "result.json", result)
    _write_json(out / "provenance.json", {"signature": signature, "pair_manifest_sha256": _sha_file(REPORT_DIR / f"{task}_pairs.csv"),
                                            "test_prediction_sha256": _sha_file(pred_path), "support_vector_rows": len(clf.support_),
                                            "test_pairs_sha256": _sha_text("\n".join(row["pair_id"] for row in test_rows))})
    print(f"{task} SVM test balanced_accuracy={metrics['balanced_accuracy']:.6f} auc={metrics['roc_auc']:.6f}", flush=True)


def _fit_rf(task: str, arrays: dict[str, dict[str, Any]], signature: str, base_seed: int) -> None:
    out = RESULT_DIR / task / "random_forest"
    model_path = MODEL_DIR / task / "random_forest.joblib"
    out.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    state_path = out / "candidate_results.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("signature") != signature:
            raise RuntimeError(f"stale RF candidate results in {out}; inspect and move before rerunning")
    else:
        state = {"signature": signature, "candidates": {}}

    x_train_raw, y_train = arrays["train"]["x"], arrays["train"]["y"]
    x_val_raw, y_val = arrays["validation"]["x"], arrays["validation"]["y"]
    scaler = StandardScaler().fit(x_train_raw)
    x_train = scaler.transform(x_train_raw)
    x_val = scaler.transform(x_val_raw)
    for leaf in (2, 8):
        key = str(leaf)
        pred_path = out / f"leaf_{leaf}_validation.npz"
        if key in state["candidates"] and pred_path.exists():
            continue
        print(f"{task} RF validation leaf={leaf}: fitting 300 trees", flush=True)
        clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=leaf, max_features="sqrt",
                                     random_state=base_seed, n_jobs=4)
        with threadpool_limits(limits=4):
            clf.fit(x_train, y_train)
        scores = clf.predict_proba(x_val)[:, 1]
        pred = (scores >= 0.5).astype(np.int8)
        metrics = _metric_summary(y_val, pred, scores)
        np.savez_compressed(pred_path, prediction=pred, score=scores, labels=y_val)
        state["candidates"][key] = metrics
        _write_json(state_path, state)
        print(f"{task} RF leaf={leaf} validation balanced_accuracy={metrics['balanced_accuracy']:.6f} auc={metrics['roc_auc']:.6f}", flush=True)

    best_leaf = max((2, 8), key=lambda leaf: _selection_key(state["candidates"][str(leaf)], leaf))
    trainval_raw = np.concatenate((arrays["train"]["x"], arrays["validation"]["x"]), axis=0)
    trainval_y = np.concatenate((y_train, y_val))
    trainval_rows = arrays["train"]["rows"] + arrays["validation"]["rows"]
    if model_path.exists():
        bundle = joblib.load(model_path)
        if bundle.get("signature") != signature:
            raise RuntimeError(f"stale final RF model in {model_path}")
        clf, final_scaler = bundle["model"], bundle["scaler"]
    else:
        final_scaler = StandardScaler().fit(trainval_raw)
        x_trainval = final_scaler.transform(trainval_raw)
        print(f"{task} RF final refit on train+validation; leaf={best_leaf}", flush=True)
        clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=best_leaf, max_features="sqrt",
                                     random_state=base_seed, n_jobs=4)
        with threadpool_limits(limits=4):
            clf.fit(x_trainval, trainval_y)
        bundle = {"signature": signature, "model": clf, "scaler": final_scaler, "best_min_samples_leaf": best_leaf,
                  "trainval_pair_ids": [r["pair_id"] for r in trainval_rows]}
        temp_model = model_path.with_suffix(".joblib.tmp")
        joblib.dump(bundle, temp_model, compress=0)
        temp_model.replace(model_path)

    x_test = final_scaler.transform(arrays["test"]["x"])
    with threadpool_limits(limits=4):
        test_scores = clf.predict_proba(x_test)[:, 1]
    test_pred = (test_scores >= 0.5).astype(np.int8)
    y_test = arrays["test"]["y"]
    test_rows = arrays["test"]["rows"]
    metrics = _metric_summary(y_test, test_pred, test_scores)
    result = {
        "task": task, "model": "RandomForestClassifier", "signature": signature,
        "selected_min_samples_leaf": best_leaf, "n_estimators": 300, "max_features": "sqrt", "n_jobs": 4,
        "selection_metric": "validation balanced accuracy; ROC AUC tie-break", "validation_candidates": state["candidates"],
        "train_pairs": len(y_train), "validation_pairs": len(y_val),
        "final_fit_pairs_train_plus_validation": len(trainval_y), "test_pairs": len(y_test),
        "test": metrics,
        "grouped_test_metrics": _group_metrics(task, test_rows, y_test, test_pred, test_scores),
        "cluster_bootstrap": _cluster_bootstrap(test_rows, y_test, test_pred, test_scores, base_seed + 302),
        "frequency_importance": "MDI omitted; see held-out grouped permutation diagnostic, exploratory only",
        "model_file": str(model_path.relative_to(ROOT)),
    }
    pred_path = out / "test_predictions.csv"
    _save_predictions(pred_path, test_rows, test_pred, test_scores)
    permutation = _frequency_permutation(task, "random_forest", clf, x_test, y_test,
                                         lambda values: clf.predict_proba(values)[:, 1], base_seed + 402)
    with (out / "frequency_permutation.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(permutation[0]))
        writer.writeheader()
        writer.writerows(permutation)
    _write_json(out / "result.json", result)
    _write_json(out / "provenance.json", {"signature": signature, "pair_manifest_sha256": _sha_file(REPORT_DIR / f"{task}_pairs.csv"),
                                            "test_prediction_sha256": _sha_file(pred_path), "test_pairs_sha256": _sha_text("\n".join(row["pair_id"] for row in test_rows))})
    print(f"{task} RF test balanced_accuracy={metrics['balanced_accuracy']:.6f} auc={metrics['roc_auc']:.6f}", flush=True)


def _write_run_summary(entries: list[dict[str, Any]], status: str, current: dict[str, str] | None = None) -> None:
    _write_json(REPORT_DIR / "run_status.json", {"status": status, "updated_epoch_seconds": time.time(), "models": entries,
                                                   "current": current,
                                                   "T2": {"status": "unavailable_missing_true_session_mapping", "accuracy": None}})


def main() -> None:
    _review_gate()
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if not (FEATURE_DIR / "features.json").exists() or not (FEATURE_DIR / "features.npy").exists():
        raise FileNotFoundError("complete A077 feature cache missing")
    feature_meta = json.loads((FEATURE_DIR / "features.json").read_text(encoding="utf-8"))
    if feature_meta.get("status") != "complete" or feature_meta.get("shape", [None])[1:] != [62, 32]:
        raise ValueError("A077 feature cache metadata is not complete or has the wrong shape")
    config_hash = _sha_text(json.dumps(cfg, sort_keys=True, ensure_ascii=False))
    features = np.load(FEATURE_DIR / "features.npy", mmap_mode="r", allow_pickle=False)
    if list(features.shape) != feature_meta["shape"]:
        raise ValueError("feature cache file and metadata shape differ")
    tasks = ("T1", "T3")
    work: list[tuple[str, list[dict[str, str]], dict[str, str], dict[str, dict[str, Any]], str]] = []
    for task in tasks:
        rows, manifest_meta = _read_pairs(task)
        arrays = _pair_arrays(rows, features)
        signature = _sha_text(json.dumps({"config_sha256": config_hash, "feature_manifest_sha256": feature_meta["trial_manifest_sha256"],
                                          "pair_manifest_sha256": manifest_meta["sha256"], "feature_config_sha256": feature_meta["features_config_sha256"]}, sort_keys=True))
        work.append((task, rows, manifest_meta, arrays, signature))

    completed = []
    _write_run_summary(completed, "running")
    start = time.time()
    torch.set_num_threads(4)
    with threadpool_limits(limits=4):
        for task, rows, manifest_meta, arrays, signature in work:
            print(f"Starting {task}: {len(rows)} pairs from shared manifest {manifest_meta['sha256']}", flush=True)
            _write_run_summary(completed, "running", {"task": task, "model": "svm", "phase": "validation_grid_and_train_validation_refit"})
            _fit_svm(task, arrays, signature, int(cfg["seed"]) + (11 if task == "T1" else 31))
            completed.append({"task": task, "model": "svm", "signature": signature, "status": "complete"})
            _write_run_summary(completed, "running")
            _write_run_summary(completed, "running", {"task": task, "model": "random_forest", "phase": "validation_grid_and_train_validation_refit"})
            _fit_rf(task, arrays, signature, int(cfg["seed"]) + (11 if task == "T1" else 31))
            completed.append({"task": task, "model": "random_forest", "signature": signature, "status": "complete"})
            _write_run_summary(completed, "running")
    _write_run_summary(completed, "complete")
    _write_json(REPORT_DIR / "RUN_COMPLETE.json", {"status": "complete", "models": completed, "T2": "unavailable_missing_true_session_mapping",
                                                    "elapsed_seconds": time.time() - start})
    print(f"A077 model runs complete in {time.time() - start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
