"""Prepare A077 metadata, leak-safe pair manifests, and the resumable PSD cache.

This script performs no classifier fit. EEG archives are read through the
repository's safe weights-only, mmap loader and converted in small batches.
"""

from __future__ import annotations

import os

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "4"

import csv
import hashlib
import itertools
import json
import math
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from eegdecoding.data import PART_NAMES, iter_records, load_part  # noqa: E402

CONFIG_PATH = ROOT / "config/a077_pair_tasks.json"
REPORT_DIR = ROOT / "reports/a077_pair_tasks"
ARTIFACT_DIR = ROOT / "artifacts/a077_pair_features"
TRIALS_PATH = REPORT_DIR / "trials.csv"
FEATURE_PATH = ARTIFACT_DIR / "features.npy"
PARTIAL_FEATURE_PATH = ARTIFACT_DIR / "features.partial.npy"
FEATURE_META_PATH = ARTIFACT_DIR / "features.json"
FEATURE_STATE_PATH = ARTIFACT_DIR / "feature_extraction_state.json"
MANIFEST_FIELDS = [
    "pair_id", "match_id", "split", "target", "target_name", "a", "b",
    "category_index", "category", "image_a", "image_b", "subject_a", "subject_b",
    "position_a", "position_b", "center", "gap",
]


def _json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _metadata_and_inventory(cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records = [asdict(row) for row in iter_records(ROOT)]
    part_order = {name: i for i, name in enumerate(PART_NAMES)}
    records.sort(key=lambda row: (part_order[row["source_file"]], int(row["source_index"])))
    for feature_row, row in enumerate(records):
        row["feature_row"] = feature_row

    fields_by_file: dict[str, Counter[str]] = {}
    per_file_subjects: dict[str, Counter[int]] = {}
    session_fields: set[str] = set()
    for filename in PART_NAMES:
        archive = load_part(ROOT / filename)
        field_counts: Counter[str] = Counter()
        subject_counts: Counter[int] = Counter()
        for item in archive["dataset"]:
            field_counts.update(str(k) for k in item.keys())
            subject_counts[int(item["subject"])] += 1
        fields_by_file[filename] = field_counts
        per_file_subjects[filename] = subject_counts
        session_fields.update(k for k in field_counts if "session" in k.lower() or "stage" in k.lower())
        del archive

    subject_counts = Counter(int(row["subject"]) for row in records)
    category_counts = Counter(int(row["label_index"]) for row in records)
    images_by_category: dict[int, set[str]] = defaultdict(set)
    subjects_by_image: dict[str, set[int]] = defaultdict(set)
    images_by_id: dict[str, int] = {}
    for row in records:
        category = int(row["label_index"])
        image = str(row["image"])
        if image in images_by_id and images_by_id[image] != category:
            raise ValueError(f"image {image} maps to multiple categories")
        images_by_id[image] = category
        images_by_category[category].add(image)
        subjects_by_image[image].add(int(row["subject"]))

    expected = int(cfg["source"]["expected_records"])
    expected_subjects = set(range(int(cfg["source"]["expected_subjects"])))
    if len(records) != expected or set(subject_counts) != expected_subjects:
        raise ValueError(f"source inventory differs from A077 contract: records={len(records)}, subjects={sorted(subject_counts)}")
    if len(images_by_id) != 4000 or len(images_by_category) != 80 or any(len(v) != 50 for v in images_by_category.values()):
        raise ValueError("expected 80 categories with 50 distinct image IDs each")
    if session_fields:
        raise ValueError(f"session/stage fields detected; stop for explicit source review: {sorted(session_fields)}")

    file_rows = []
    for filename in PART_NAMES:
        subset = [row for row in records if row["source_file"] == filename]
        file_rows.append({
            "file": filename,
            "bytes": (ROOT / filename).stat().st_size,
            "records": len(subset),
            "subjects": sorted(per_file_subjects[filename]),
            "metadata_fields": sorted(fields_by_file[filename]),
        })
    inventory = {
        "source": "local EEG-ImageNet v1 archives loaded using weights_only=True, mmap=True",
        "files": file_rows,
        "records": len(records),
        "subjects": len(subject_counts),
        "records_by_subject": {str(k): int(v) for k, v in sorted(subject_counts.items())},
        "categories": len(category_counts),
        "records_by_category": {str(k): int(v) for k, v in sorted(category_counts.items())},
        "unique_images": len(images_by_id),
        "images_per_category": {str(k): len(v) for k, v in sorted(images_by_category.items())},
        "subjects_per_image": {str(k): int(v) for k, v in sorted(Counter(len(v) for v in subjects_by_image.values()).items())},
        "session_or_stage_fields": sorted(session_fields),
        "session_mapping_available": False,
        "additional_24000_records_located": False,
        "trial_manifest_sha256": _sha_text(json.dumps(records, sort_keys=True, ensure_ascii=False, default=str)),
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with TRIALS_PATH.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    _json_write(REPORT_DIR / "source_inventory.json", inventory)
    return records, inventory


def _image_splits(cfg: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, list[int]]:
    labels = sorted({int(row["label_index"]) for row in records})
    if len(labels) != 80:
        raise ValueError(f"expected 80 image categories, found {len(labels)}")
    rng = np.random.default_rng(int(cfg["split"]["seed"]))
    perm = [int(x) for x in rng.permutation(labels)]
    sizes = cfg["split"]["counts"]
    if sum(int(sizes[name]) for name in ("train", "validation", "test")) != len(labels):
        raise ValueError("configured category split sizes do not sum to available categories")
    n_train, n_val = int(sizes["train"]), int(sizes["validation"])
    result = {"train": sorted(perm[:n_train]), "validation": sorted(perm[n_train:n_train + n_val]), "test": sorted(perm[n_train + n_val:])}
    if tuple(len(result[k]) for k in ("train", "validation", "test")) != (48, 16, 16):
        raise AssertionError("A077 category split must be 48/16/16")
    label_names: dict[str, str] = {}
    for row in records:
        key = str(int(row["label_index"]))
        value = str(row["label"])
        if key in label_names and label_names[key] != value:
            raise ValueError(f"label index {key} maps to multiple category names")
        label_names[key] = value
    _json_write(REPORT_DIR / "class_split.json", {
        "seed": int(cfg["split"]["seed"]), "unit": cfg["split"]["unit"],
        "label_index_to_category": label_names,
        "categories_by_split": result,
        "counts": {key: len(value) for key, value in result.items()},
    })
    return result


def _balanced_category_pick(candidates: dict[int, list[Any]], target: int, rng: np.random.Generator) -> list[tuple[int, Any]]:
    keys = sorted(candidates)
    for key in keys:
        rng.shuffle(candidates[key])
    rng.shuffle(keys)
    picked: list[tuple[int, Any]] = []
    cursor = {key: 0 for key in keys}
    while len(picked) < target:
        made_progress = False
        for key in keys:
            if cursor[key] < len(candidates[key]):
                picked.append((key, candidates[key][cursor[key]]))
                cursor[key] += 1
                made_progress = True
                if len(picked) == target:
                    break
        if not made_progress:
            raise ValueError(f"only {len(picked)} eligible matched units for target {target}")
    return picked


def _balanced_group_pick(groups: dict[tuple[int, int], list[int]], target: int, rng: np.random.Generator) -> list[tuple[tuple[int, int], int]]:
    keys = list(groups)
    rng.shuffle(keys)
    for key in keys:
        rng.shuffle(groups[key])
    cursor = {key: 0 for key in keys}
    picked: list[tuple[tuple[int, int], int]] = []
    while len(picked) < target:
        made_progress = False
        for key in keys:
            if cursor[key] < len(groups[key]):
                picked.append((key, groups[key][cursor[key]]))
                cursor[key] += 1
                made_progress = True
                if len(picked) == target:
                    break
        if not made_progress:
            raise ValueError(f"only {len(picked)} eligible matched centers for target {target}")
    return picked


def _make_pair_row(task: str, split: str, match_id: str, target: int, a: dict[str, Any], b: dict[str, Any], center: float | None, gap: int | None) -> dict[str, Any]:
    return {
        "pair_id": "", "match_id": match_id, "split": split, "target": int(target),
        "target_name": ("same_person" if task == "T1" and target else "different_people" if task == "T1" else "adjacent" if target else "gap_7"),
        "a": int(a["feature_row"]), "b": int(b["feature_row"]),
        "category_index": int(a["label_index"]), "category": str(a["label"]),
        "image_a": str(a["image"]), "image_b": str(b["image"]),
        "subject_a": int(a["subject"]), "subject_b": int(b["subject"]),
        "position_a": int(a["position_in_class_block"]), "position_b": int(b["position_in_class_block"]),
        "center": "" if center is None else float(center), "gap": "" if gap is None else int(gap),
    }


def _make_t1(records: list[dict[str, Any]], splits: dict[str, list[int]], cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_label_image: dict[tuple[int, str], dict[int, dict[str, Any]]] = defaultdict(dict)
    by_category_images: dict[int, set[str]] = defaultdict(set)
    for row in records:
        label, image, subject = int(row["label_index"]), str(row["image"]), int(row["subject"])
        if subject in by_label_image[(label, image)]:
            raise ValueError(f"duplicate subject/category/image record: {subject}/{label}/{image}")
        by_label_image[(label, image)][subject] = row
        by_category_images[label].add(image)

    pair_rows: list[dict[str, Any]] = []
    capacity: dict[str, int] = {}
    for split_ix, split in enumerate(("train", "validation", "test")):
        rng = np.random.default_rng(int(cfg["seed"]) + 101 + split_ix)
        candidates: dict[int, list[tuple[str, str]]] = {}
        for label in splits[split]:
            images = sorted(by_category_images[label])
            eligible = []
            for image_a, image_b in itertools.combinations(images, 2):
                subs_a, subs_b = set(by_label_image[(label, image_a)]), set(by_label_image[(label, image_b)])
                if subs_a & subs_b and any(sa != sb for sa in subs_a for sb in subs_b):
                    eligible.append((image_a, image_b))
            candidates[label] = eligible
        capacity[split] = sum(len(v) for v in candidates.values())
        pair_cap = int(cfg["pair_tasks"]["T1_same_person"]["pair_caps_by_split"][split])
        units = pair_cap // 2
        selected = _balanced_category_pick(candidates, units, rng)
        pos_usage: Counter[int] = Counter()
        neg_usage: Counter[int] = Counter()
        for unit_index, (label, (image_a, image_b)) in enumerate(selected):
            rows_a = by_label_image[(label, image_a)]
            rows_b = by_label_image[(label, image_b)]
            common = sorted(set(rows_a) & set(rows_b))
            least_used = min(pos_usage[s] for s in common)
            pos_choices = [s for s in common if pos_usage[s] == least_used]
            subject_pos = int(rng.choice(pos_choices))
            pos_usage[subject_pos] += 1

            cross = [(sa, sb) for sa in rows_a for sb in rows_b if sa != sb]
            best = min(neg_usage[sa] + neg_usage[sb] for sa, sb in cross)
            neg_choices = [(sa, sb) for sa, sb in cross if neg_usage[sa] + neg_usage[sb] == best]
            subject_a, subject_b = (int(x) for x in neg_choices[int(rng.integers(len(neg_choices)))])
            neg_usage[subject_a] += 1
            neg_usage[subject_b] += 1

            match_id = f"T1-{split}-{unit_index:05d}"
            pair_rows.append(_make_pair_row("T1", split, match_id, 1, rows_a[subject_pos], rows_b[subject_pos], None, None))
            pair_rows.append(_make_pair_row("T1", split, match_id, 0, rows_a[subject_a], rows_b[subject_b], None, None))

    for pair_id, row in enumerate(pair_rows):
        row["pair_id"] = f"T1-{pair_id:06d}"
    _write_pairs(REPORT_DIR / "T1_pairs.csv", pair_rows)
    summary = _summarize_pairs("T1", pair_rows, capacity, eligible_units_by_split={k: v for k, v in capacity.items()})
    return pair_rows, summary


def _make_t3(records: list[dict[str, Any]], splits: dict[str, list[int]], cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_subject_category_position: dict[tuple[int, int], dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in records:
        key = (int(row["subject"]), int(row["label_index"]))
        position = int(row["position_in_class_block"])
        if position in by_subject_category_position[key]:
            raise ValueError(f"duplicate position within subject/category: {key}/{position}")
        by_subject_category_position[key][position] = row

    pair_rows: list[dict[str, Any]] = []
    capacity: dict[str, int] = {}
    range_cfg = cfg["pair_tasks"]["T3_adjacent_within_person_category"]["eligible_center_k"]
    for split_ix, split in enumerate(("train", "validation", "test")):
        rng = np.random.default_rng(int(cfg["seed"]) + 201 + split_ix)
        groups: dict[tuple[int, int], list[int]] = {}
        for label in splits[split]:
            for (subject, category), position_rows in by_subject_category_position.items():
                if category != label:
                    continue
                centers = [k for k in range(int(range_cfg[0]), int(range_cfg[1]) + 1)
                           if all(p in position_rows for p in (k - 3, k, k + 1, k + 4))]
                if centers:
                    groups[(label, subject)] = centers
        n_eligible_units = sum(len(v) for v in groups.values())
        capacity[split] = n_eligible_units
        pair_cap = int(cfg["pair_tasks"]["T3_adjacent_within_person_category"]["pair_caps_by_split"][split])
        selected = _balanced_group_pick(groups, pair_cap // 2, rng)
        for unit_index, ((label, subject), center_k) in enumerate(selected):
            position_rows = by_subject_category_position[(subject, label)]
            pos_a, pos_b = position_rows[center_k], position_rows[center_k + 1]
            neg_a, neg_b = position_rows[center_k - 3], position_rows[center_k + 4]
            match_id = f"T3-{split}-{unit_index:05d}"
            center = center_k + 0.5
            pair_rows.append(_make_pair_row("T3", split, match_id, 1, pos_a, pos_b, center, 1))
            pair_rows.append(_make_pair_row("T3", split, match_id, 0, neg_a, neg_b, center, 7))

    for pair_id, row in enumerate(pair_rows):
        row["pair_id"] = f"T3-{pair_id:06d}"
    _write_pairs(REPORT_DIR / "T3_pairs.csv", pair_rows)
    summary = _summarize_pairs("T3", pair_rows, capacity, eligible_units_by_split=capacity)
    return pair_rows, summary


def _write_pairs(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _summarize_pairs(task: str, rows: list[dict[str, Any]], capacity: dict[str, int], eligible_units_by_split: dict[str, int]) -> dict[str, Any]:
    result: dict[str, Any] = {"task": task, "eligible_matched_units_by_split": {k: int(v) for k, v in capacity.items()}, "splits": {}}
    for split in ("train", "validation", "test"):
        subset = [row for row in rows if row["split"] == split]
        positives = [row for row in subset if int(row["target"]) == 1]
        negatives = [row for row in subset if int(row["target"]) == 0]
        trials = {int(row[x]) for row in subset for x in ("a", "b")}
        images = {str(row[x]) for row in subset for x in ("image_a", "image_b")}
        subjects = {int(row[x]) for row in subset for x in ("subject_a", "subject_b")}
        categories = {int(row["category_index"]) for row in subset}
        result["splits"][split] = {
            "pairs": len(subset), "positive_pairs": len(positives), "negative_pairs": len(negatives),
            "eligible_matched_units": int(eligible_units_by_split[split]),
            "selected_independent_trials": len(trials), "selected_unique_images": len(images),
            "selected_subjects": sorted(subjects), "selected_subject_count": len(subjects),
            "selected_categories": len(categories),
            "subject_pair_counts": {"same_subject_positive_pairs": len(positives), "cross_subject_negative_pairs": len(negatives)},
        }
        if len(positives) != len(negatives) or not subset:
            raise AssertionError(f"{task}/{split}: pair classes are not present and balanced")
    _audit_pairs(task, rows)
    return result


def _audit_pairs(task: str, rows: list[dict[str, Any]]) -> None:
    seen_pair: set[tuple[str, int, int, int]] = set()
    match_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    split_images: dict[str, set[str]] = defaultdict(set)
    split_trials: dict[str, set[int]] = defaultdict(set)
    split_categories: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        a, b = int(row["a"]), int(row["b"])
        fingerprint = (str(row["split"]), min(a, b), max(a, b), int(row["target"]))
        if fingerprint in seen_pair:
            raise AssertionError(f"duplicate pair: {fingerprint}")
        seen_pair.add(fingerprint)
        match_groups[str(row["match_id"])].append(row)
        split_images[str(row["split"])].update((str(row["image_a"]), str(row["image_b"])))
        split_trials[str(row["split"])].update((a, b))
        split_categories[str(row["split"])].add(int(row["category_index"]))
        if task == "T1":
            if row["image_a"] == row["image_b"] or not row["category"]:
                raise AssertionError("T1 requires two distinct images in one category")
            same_person = int(row["subject_a"]) == int(row["subject_b"])
            if same_person != bool(int(row["target"])):
                raise AssertionError("T1 label does not match subject identity")
        else:
            if int(row["subject_a"]) != int(row["subject_b"]) or int(row["position_b"]) - int(row["position_a"]) != int(row["gap"]):
                raise AssertionError("T3 same-subject or source-gap condition failed")
            if int(row["gap"]) == 1 and int(row["target"]) != 1:
                raise AssertionError("T3 adjacent target mismatch")
            if int(row["gap"]) == 7 and int(row["target"]) != 0:
                raise AssertionError("T3 gap-7 target mismatch")
            midpoint = (int(row["position_a"]) + int(row["position_b"])) / 2
            if not math.isclose(midpoint, float(row["center"]), abs_tol=1e-12):
                raise AssertionError("T3 matched center mismatch")

    for match_id, group in match_groups.items():
        if len(group) != 2 or {int(row["target"]) for row in group} != {0, 1}:
            raise AssertionError(f"bad matched group: {match_id}")
        left, right = group
        if int(left["category_index"]) != int(right["category_index"]):
            raise AssertionError(f"matched category mismatch: {match_id}")
        if task == "T1" and {str(left["image_a"]), str(left["image_b"])} != {str(right["image_a"]), str(right["image_b"])}:
            raise AssertionError(f"T1 positive/negative image pair mismatch: {match_id}")
        if task == "T3" and not math.isclose(float(left["center"]), float(right["center"]), abs_tol=1e-12):
            raise AssertionError(f"T3 positive/negative center mismatch: {match_id}")

    splits = ("train", "validation", "test")
    for i, left in enumerate(splits):
        for right in splits[i + 1:]:
            if split_images[left] & split_images[right] or split_trials[left] & split_trials[right] or split_categories[left] & split_categories[right]:
                raise AssertionError(f"{task} split overlap: {left}/{right}")
    if task == "T1":
        for split in splits:
            used = {int(row[k]) for row in rows if row["split"] == split for k in ("subject_a", "subject_b")}
            if used != set(range(16)):
                raise AssertionError(f"T1 {split} does not include all 16 subjects: {sorted(used)}")


def _feature_batch(raw: np.ndarray, window: np.ndarray) -> np.ndarray:
    values = np.asarray(raw, dtype=np.float64)
    values = values - values.mean(axis=-1, keepdims=True)
    spectrum = np.fft.rfft(values * window, n=400, axis=-1)
    psd = np.square(np.abs(spectrum)) / (1000.0 * np.square(window).sum())
    psd[..., 1:-1] *= 2.0
    features = np.log10(np.maximum(psd[..., 1:33], 1e-30)).astype(np.float32)
    if features.shape[1:] != (62, 32) or not np.isfinite(features).all():
        raise ValueError("invalid log-PSD feature batch")
    return features


def _extract_features(records: list[dict[str, Any]], inventory: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    cfg_hash = _sha_text(json.dumps(cfg["features"], sort_keys=True))
    manifest_hash = str(inventory["trial_manifest_sha256"])
    input_sig = {item["file"]: {"bytes": item["bytes"], "records": item["records"]} for item in inventory["files"]}
    expected_meta = {"shape": [len(records), 62, 32], "dtype": "float32", "features_config_sha256": cfg_hash,
                     "trial_manifest_sha256": manifest_hash, "source_signature": input_sig}
    if FEATURE_PATH.exists():
        if FEATURE_META_PATH.exists():
            meta = json.loads(FEATURE_META_PATH.read_text(encoding="utf-8"))
            for key, value in expected_meta.items():
                if meta.get(key) != value:
                    raise RuntimeError(f"existing feature cache metadata mismatch for {key}")
            cache = np.load(FEATURE_PATH, mmap_mode="r", allow_pickle=False)
            if list(cache.shape) != expected_meta["shape"] or cache.dtype != np.float32:
                raise RuntimeError("completed feature cache shape/dtype mismatch")
            return meta
        # Recover the narrow interruption window after the partial file was
        # renamed but before its completion metadata was committed.
        if not FEATURE_STATE_PATH.exists() or PARTIAL_FEATURE_PATH.exists():
            raise RuntimeError("completed feature file exists without matching recovery state; refusing to reuse")
        state = json.loads(FEATURE_STATE_PATH.read_text(encoding="utf-8"))
        if state.get("expected_meta") != expected_meta or state.get("next_source_index") != {item["file"]: item["records"] for item in inventory["files"]}:
            raise RuntimeError("renamed feature file is not complete according to its recovery state")
        cache = np.load(FEATURE_PATH, mmap_mode="r", allow_pickle=False)
        if list(cache.shape) != expected_meta["shape"] or cache.dtype != np.float32:
            raise RuntimeError("recovered feature cache shape/dtype mismatch")
        for start in range(0, len(cache), 1024):
            if not np.isfinite(cache[start:start + 1024]).all():
                raise RuntimeError(f"recovered feature cache contains nonfinite values near row {start}")
        meta = dict(expected_meta)
        meta.update({"status": "complete", "recovered_after_atomic_rename": True,
                     "source": "safe mmap archive loader", "crop_samples": [40, 440], "sampling_rate_hz": 1000,
                     "window": "periodic Hann, 400 samples", "frequency_hz": [2.5 * i for i in range(1, 33)],
                     "demeaned_per_trial_channel": True, "subject_or_position_inputs": False,
                     "feature_file_bytes": FEATURE_PATH.stat().st_size})
        _json_write(FEATURE_META_PATH, meta)
        FEATURE_STATE_PATH.unlink()
        return meta

    if FEATURE_STATE_PATH.exists():
        state = json.loads(FEATURE_STATE_PATH.read_text(encoding="utf-8"))
        if state.get("expected_meta") != expected_meta or not PARTIAL_FEATURE_PATH.exists():
            raise RuntimeError("partial feature cache state does not match current inputs; inspect before recovery")
        next_index = {key: int(value) for key, value in state["next_source_index"].items()}
    else:
        if PARTIAL_FEATURE_PATH.exists():
            raise RuntimeError("partial feature file has no state file; refusing to overwrite")
        cache = np.lib.format.open_memmap(PARTIAL_FEATURE_PATH, mode="w+", dtype=np.float32, shape=tuple(expected_meta["shape"]))
        cache.flush()
        del cache
        next_index = {filename: 0 for filename in PART_NAMES}
        _json_write(FEATURE_STATE_PATH, {"expected_meta": expected_meta, "next_source_index": next_index, "status": "running"})

    row_by_part_source = {name: {} for name in PART_NAMES}
    for row in records:
        row_by_part_source[row["source_file"]][int(row["source_index"])] = int(row["feature_row"])
    window = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(400, dtype=np.float64) / 400.0)
    torch.set_num_threads(4)
    batch_size = 64
    start_time = time.time()
    with threadpool_limits(limits=4):
        output = np.load(PARTIAL_FEATURE_PATH, mmap_mode="r+", allow_pickle=False)
        for filename in PART_NAMES:
            archive = load_part(ROOT / filename)
            dataset = archive["dataset"]
            if len(dataset) != len(row_by_part_source[filename]):
                raise ValueError(f"{filename}: source index map does not cover each record")
            cursor = next_index[filename]
            while cursor < len(dataset):
                end = min(cursor + batch_size, len(dataset))
                eeg_rows = []
                feature_rows = []
                for source_index in range(cursor, end):
                    item = dataset[source_index]
                    row_index = row_by_part_source[filename][source_index]
                    trial = records[row_index]
                    if (int(item["subject"]) != int(trial["subject"]) or str(item["label"]) != str(trial["label"])
                            or str(item["image"]) != str(trial["image"])):
                        raise ValueError(f"source record identity mismatch at {filename}:{source_index}")
                    eeg = item["eeg_data"]
                    if tuple(eeg.shape) != (62, 501):
                        raise ValueError(f"unexpected raw EEG shape at {filename}:{source_index}: {tuple(eeg.shape)}")
                    eeg_rows.append(eeg[:, 40:440].numpy())
                    feature_rows.append(row_index)
                raw_batch = np.stack(eeg_rows, axis=0)
                output[np.asarray(feature_rows, dtype=np.int64)] = _feature_batch(raw_batch, window)
                output.flush()
                cursor = end
                next_index[filename] = cursor
                _json_write(FEATURE_STATE_PATH, {"expected_meta": expected_meta, "next_source_index": next_index, "status": "running"})
                if cursor == len(dataset) or cursor % 1024 < batch_size:
                    print(f"features {filename}: {cursor}/{len(dataset)} source rows", flush=True)
            del archive
        output.flush()
        del output

    PARTIAL_FEATURE_PATH.replace(FEATURE_PATH)
    meta = dict(expected_meta)
    meta.update({
        "status": "complete", "source": "safe mmap archive loader", "crop_samples": [40, 440],
        "sampling_rate_hz": 1000, "window": "periodic Hann, 400 samples", "frequency_hz": [2.5 * i for i in range(1, 33)],
        "demeaned_per_trial_channel": True, "subject_or_position_inputs": False,
        "completed_utc_epoch_seconds": time.time(), "elapsed_seconds": time.time() - start_time,
        "feature_file_bytes": FEATURE_PATH.stat().st_size,
    })
    _json_write(FEATURE_META_PATH, meta)
    if FEATURE_STATE_PATH.exists():
        FEATURE_STATE_PATH.unlink()
    return meta


def _write_ready(inventory: dict[str, Any], splits: dict[str, list[int]], t1: dict[str, Any], t3: dict[str, Any], feature_meta: dict[str, Any], cfg: dict[str, Any]) -> None:
    cap = cfg["pair_tasks"]["T1_same_person"]["pair_caps_by_split"]
    class_count = {k: len(v) for k, v in splits.items()}
    subject_count = inventory["records_by_subject"]
    file_lines = ["| Archive | Bytes | Trials | Subjects | Metadata fields |", "|---|---:|---:|---:|---|"]
    for item in inventory["files"]:
        file_lines.append(f"| `{item['file']}` | {item['bytes']:,} | {item['records']:,} | {len(item['subjects'])} | `{', '.join(item['metadata_fields'])}` |")
    subject_lines = ["| Subject | Trials |", "|---:|---:|"]
    subject_lines.extend(f"| {subject} | {count:,} |" for subject, count in subject_count.items())

    def task_table(summary: dict[str, Any]) -> list[str]:
        lines = ["| Split | Pairs | Positive | Negative | Matched units | Trials | Images | Subjects | Categories |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for split in ("train", "validation", "test"):
            value = summary["splits"][split]
            lines.append(f"| {split} | {value['pairs']:,} | {value['positive_pairs']:,} | {value['negative_pairs']:,} | {value['eligible_matched_units']:,} | {value['selected_independent_trials']:,} | {value['selected_unique_images']:,} | {value['selected_subject_count']} | {value['selected_categories']} |")
        return lines

    t2_note = (
        "The official records expose `eeg_data`, granularity, subject, label, and image fields, but no "
        "independent session or acquisition-stage identifier. The 30/20 class-block positions are the "
        "published train/test ordering, not verified recording sessions. T2 is therefore not constructed, "
        "fitted, or assigned an accuracy."
    )
    body = [
        "# A077 Pair-Task Preparation Report",
        "",
        "Status: `PRE_RUN_READY`. No classifier is fitted by this preparation step. The model runner requires an explicit `Decision: PASS` in `PRE_RUN_REVIEW.md`.",
        "",
        "## Source inventory",
        "",
        f"The inventory contains {inventory['records']:,} trials, {inventory['subjects']} subjects, {inventory['categories']} categories, and {inventory['unique_images']:,} unique image IDs. Archives are loaded with `weights_only=True, mmap=True`.",
        "",
        *file_lines,
        "",
        "| Subject | Trials |", *subject_lines[1:],
        "",
        f"Each category contains {min(inventory['images_per_category'].values())}--{max(inventory['images_per_category'].values())} images; each image is represented for {inventory['subjects_per_image']} subjects. Raw EEG has shape 62 x 501, and features use samples 40:440.",
        "",
        "## Split and features",
        "",
        f"Within each of the 80 categories, image IDs are split deterministically with seed {cfg['seed']} into 48/16/16 train, validation, and test groups; `class_split.json` records the assignment. Features have shape {feature_meta['shape']} in float32 and contain 32 Hann-window log10-PSD bins from 2.5 to 80 Hz. Subject, source index, and audit metadata are retained for grouping but are not model inputs.",
        "",
        "## T1: same-person discrimination",
        "",
        "Positive pairs share subject identity within the same image ID; negative pairs use different subjects under the same image. Image-level groups remain disjoint across splits, and all 16 subjects are represented.",
        "",
        *task_table(t1),
        "",
        "## T2: same-session discrimination",
        "",
        t2_note,
        "",
        "| Split | Pairs | Session field | Accuracy |", "|---|---:|---|---|",
        "| train/validation/test | 0 | unavailable | not reported |",
        "",
        "## T3: temporal-position discrimination",
        "",
        "Positive pairs use adjacent class-block positions `(k, k+1)` and negatives use separated positions `(k-3, k+4)` under matched grouping constraints. This task explicitly probes the ordering effect; class-block position is only a temporal proxy, not a verified session label.",
        "",
        *task_table(t3),
        "",
        "## Outputs and safeguards",
        "",
        "- Inventory: `source_inventory.json` and `trials.csv`; split contract: `class_split.json`.",
        "- Pair tables: `T1_pairs.csv` and `T3_pairs.csv`; the model runner consumes these tables directly.",
        "- Feature cache: `../../artifacts/a077_pair_features/features.npy` with `features.json`; the runner memory-maps this cache.",
        "- Reproducible source: `../../scripts/a077_prepare_pair_tasks.py`, `../../scripts/a077_run_models.py`, and `../../config/a077_pair_tasks.json`.",
        "- Group checks, bootstrap grouping, support thresholds, and split-disjointness checks must pass before fitting. The 70% threshold is a predeclared interpretation threshold, not a property of the test distribution.",
        "- All available subjects are included; this is not a subject-0-only experiment.",
        "",
        f"Feature-cache status: `{feature_meta['status']}`; size: {feature_meta['feature_file_bytes']:,} bytes.",
        "",
    ]
    (REPORT_DIR / "PRE_RUN_READY.md").write_text("\n".join(body), encoding="utf-8")


def main() -> None:
    started = time.time()
    torch.set_num_threads(4)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = _load_config()
    print("A077 metadata inventory (safe mmap; no model fitting)", flush=True)
    records, inventory = _metadata_and_inventory(cfg)
    splits = _image_splits(cfg, records)
    print(f"records={len(records)} subjects={inventory['subjects']} categories={inventory['categories']}", flush=True)
    t1_rows, t1_summary = _make_t1(records, splits, cfg)
    t3_rows, t3_summary = _make_t3(records, splits, cfg)
    if len(t1_rows) != sum(int(v) for v in cfg["pair_tasks"]["T1_same_person"]["pair_caps_by_split"].values()):
        raise AssertionError("T1 pair caps not met")
    if len(t3_rows) != sum(int(v) for v in cfg["pair_tasks"]["T3_adjacent_within_person_category"]["pair_caps_by_split"].values()):
        raise AssertionError("T3 pair caps not met")
    _json_write(REPORT_DIR / "pair_counts.json", {"T1": t1_summary, "T2": {"status": "unavailable_missing_true_session_mapping", "pair_counts": {"train": 0, "validation": 0, "test": 0}, "accuracy": None}, "T3": t3_summary})
    print("Extracting 62x32 log-PSD features in batches (no classifier fit)", flush=True)
    feature_meta = _extract_features(records, inventory, cfg)
    _write_ready(inventory, splits, t1_summary, t3_summary, feature_meta, cfg)
    print(json.dumps({"status": "PRE_RUN_READY", "t1_pairs": len(t1_rows), "t3_pairs": len(t3_rows), "feature_cache": str(FEATURE_PATH), "elapsed_seconds": time.time() - started}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main()
