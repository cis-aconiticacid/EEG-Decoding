"""Project-relative, verified access to the EEG-ImageNet v1 archives."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"
PART_NAMES = ("EEG-ImageNet_1.pth", "EEG-ImageNet_2.pth")
ASSETS = {
    "EEG-ImageNet_1.pth": {
        "url": "https://cloud.tsinghua.edu.cn/d/d812f7d1fc474b14bbd0/files/?p=%2FEEG-ImageNet_1.pth&dl=1",
        "bytes": 7_944_350_936,
        "sha256": "57e187c0515587f2b7283f41dc439df4ba8ac5fe106b8c3e6ee850d2a9a6d8d9",
    },
    "EEG-ImageNet_2.pth": {
        "url": "https://cloud.tsinghua.edu.cn/d/d812f7d1fc474b14bbd0/files/?p=%2FEEG-ImageNet_2.pth&dl=1",
        "bytes": 7_931_918_744,
        "sha256": "0e54b706d5f06a56bc590fa5dabed06d640ff4e28fed97db766b012ce24c452e",
    },
}


def default_data_path(name: str) -> Path:
    """Return the canonical project data path for a known archive name."""
    if name not in ASSETS:
        raise ValueError(f"Unknown EEG archive: {name}")
    return DATA_ROOT / name


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _verified_sidecar(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".verified.json")


def _verify(path: Path, asset: dict[str, object]) -> None:
    expected_size = int(asset["bytes"])
    expected_sha = str(asset["sha256"])
    if path.stat().st_size != expected_size:
        raise ValueError(
            f"{path} has {path.stat().st_size} bytes; expected {expected_size}. "
            "The existing file was not overwritten."
        )
    sidecar = _verified_sidecar(path)
    if sidecar.exists():
        try:
            record = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            record = {}
        if record.get("bytes") == expected_size and record.get("sha256") == expected_sha:
            return
    actual_sha = _sha256(path)
    if actual_sha != expected_sha:
        raise ValueError(f"SHA-256 mismatch for {path}; the file was not overwritten")
    sidecar.write_text(
        json.dumps({"bytes": expected_size, "sha256": expected_sha}, indent=2) + "\n",
        encoding="utf-8",
    )


def ensure_archive(name: str) -> Path:
    """Return a verified archive, downloading it into ``data/`` when absent."""
    if name not in ASSETS:
        raise ValueError(f"Unknown EEG archive: {name}")
    asset = ASSETS[name]
    path = default_data_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        _verify(path, asset)
        return path
    partial = path.with_suffix(path.suffix + ".partial")
    if partial.exists():
        partial.unlink()
    request = urllib.request.Request(
        str(asset["url"]), headers={"User-Agent": "eegdecoding-dataset-loader/1.0"}
    )
    digest = hashlib.sha256()
    total = 0
    try:
        with urllib.request.urlopen(request) as response, partial.open("xb") as output:
            while block := response.read(8 << 20):
                output.write(block)
                digest.update(block)
                total += len(block)
        if total != int(asset["bytes"]):
            raise ValueError(f"Downloaded {total} bytes for {name}; expected {asset['bytes']}")
        if digest.hexdigest() != str(asset["sha256"]):
            raise ValueError(f"Downloaded SHA-256 mismatch for {name}")
        os.replace(partial, path)
        _verified_sidecar(path).write_text(
            json.dumps({"bytes": total, "sha256": digest.hexdigest()}, indent=2) + "\n",
            encoding="utf-8",
        )
    except Exception:
        if partial.exists():
            partial.unlink()
        raise
    return path


def _resolve_archive_path(path: str | Path) -> Path:
    """Resolve known archive names to the project ``data/`` directory."""
    candidate = Path(path)
    if candidate.name in ASSETS:
        return ensure_archive(candidate.name)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate


def _legacy_numpy_scalar(*args: object) -> object:
    return np._core.multiarray.scalar(*args)


_legacy_numpy_scalar.__module__ = "numpy.core.multiarray"
_legacy_numpy_scalar.__name__ = "scalar"
_legacy_numpy_scalar.__qualname__ = "scalar"


def _numpy_safe_globals() -> list[object]:
    return [
        _legacy_numpy_scalar,
        np._core.multiarray.scalar,
        np.dtype,
        type(np.dtype(np.int64)),
        type(np.dtype(np.float64)),
        type(np.dtype(np.str_)),
    ]


def load_part(path: str | Path) -> dict:
    """Load an archive with safe unpickling and memory-mapped storages."""
    resolved = _resolve_archive_path(path)
    with torch.serialization.safe_globals(_numpy_safe_globals()):
        return torch.load(resolved, map_location="cpu", weights_only=True, mmap=True)


@dataclass(frozen=True)
class TrialRecord:
    trial_id: str
    source_file: str
    source_index: int
    subject: int
    label: str
    label_index: int
    raw_granularity: str
    canonical_granularity: str
    granularity_mismatch: bool
    fine_task: str
    image: str
    class_block_index: int
    position_in_class_block: int
    split: str
    n_channels: int
    n_samples: int
    dtype: str


def iter_records(root: str | Path | None = None) -> Iterator[TrialRecord]:
    """Yield manifest rows without loading all EEG tensors into RAM."""
    data_root = DATA_ROOT if root is None else Path(root)
    reference_labels: list[str] | None = None
    for part_name in PART_NAMES:
        loaded = load_part(data_root / part_name)
        labels = list(loaded["labels"])
        if reference_labels is None:
            reference_labels = labels
        elif labels != reference_labels:
            raise ValueError(f"{part_name}: label list differs from the first archive")
        by_subject: dict[int, list[tuple[int, dict]]] = {}
        for source_index, item in enumerate(loaded["dataset"]):
            by_subject.setdefault(int(item["subject"]), []).append((source_index, item))
        for subject, rows in sorted(by_subject.items()):
            block_index = -1
            previous_label: str | None = None
            position = -1
            seen_labels: set[str] = set()
            for source_index, item in rows:
                label = str(item["label"])
                if label != previous_label:
                    if previous_label is not None and position != 49:
                        raise ValueError(f"Subject {subject}: incomplete class block")
                    if label in seen_labels:
                        raise ValueError(f"Subject {subject}: non-contiguous class block {label}")
                    seen_labels.add(label)
                    block_index += 1
                    position = 0
                    previous_label = label
                else:
                    position += 1
                eeg = item["eeg_data"]
                label_index = labels.index(label)
                raw_granularity = str(item["granularity"])
                canonical_granularity = "fine" if label_index < 40 else "coarse"
                stable = f"v1|s{subject:02d}|{label}|p{position:02d}|{item['image']}"
                trial_hash = hashlib.sha256(stable.encode("utf-8")).hexdigest()[:16]
                yield TrialRecord(
                    trial_id=f"v1-s{subject:02d}-{trial_hash}",
                    source_file=part_name,
                    source_index=source_index,
                    subject=subject,
                    label=label,
                    label_index=label_index,
                    raw_granularity=raw_granularity,
                    canonical_granularity=canonical_granularity,
                    granularity_mismatch=raw_granularity != canonical_granularity,
                    fine_task=f"fine{label_index // 8}" if canonical_granularity == "fine" else "",
                    image=str(item["image"]),
                    class_block_index=block_index,
                    position_in_class_block=position,
                    split="train" if position < 24 else "val" if position < 30 else "test",
                    n_channels=int(eeg.shape[0]),
                    n_samples=int(eeg.shape[1]),
                    dtype=str(eeg.dtype),
                )


def write_manifest(root: str | Path | None, output: str | Path) -> dict[str, object]:
    """Generate a trial manifest under ignored local data storage."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[tuple[int, str]] = Counter()
    records = 0
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(TrialRecord.__dataclass_fields__))
        writer.writeheader()
        for record in iter_records(root):
            writer.writerow(record.__dict__)
            counts[(record.subject, record.split)] += 1
            records += 1
    return {
        "records": records,
        "sha256": _sha256(output),
        "split_counts": {f"s{s:02d}/{split}": n for (s, split), n in sorted(counts.items())},
    }


def ensure_trial_manifest() -> Path:
    """Return the generated shared manifest, creating it from the archives if needed."""
    path = DATA_ROOT / "generated" / "trial_manifest.csv"
    if not path.exists():
        write_manifest(DATA_ROOT, path)
    return path
