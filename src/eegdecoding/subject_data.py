"""Subject-level EEG loading with prepared-array acceleration and raw-archive fallback."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .data import PROJECT_ROOT, load_part


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_subject_data(cfg: dict[str, Any]) -> dict[str, Any]:
    """Load one subject from optional prepared arrays or the verified raw archive."""
    crop_start, crop_end = map(int, cfg.get("crop", [40, 440]))
    subject_id = int(cfg.get("subject", 0))
    waveform_path = _project_path(cfg.get("waveform", "data/prepared/EEG-ImageNet_1_waveform.npy"))
    metadata_path = _project_path(cfg.get("metadata", "data/prepared/EEG-ImageNet_1_metadata.npz"))
    if waveform_path.exists() and metadata_path.exists():
        waveform = np.load(waveform_path, mmap_mode="r", allow_pickle=False)
        metadata = np.load(metadata_path, allow_pickle=False)
        selected = np.flatnonzero(metadata["subject"] == subject_id)
        x = torch.from_numpy(np.asarray(waveform[selected], dtype=np.float32).copy())
        labels = metadata["label_index"][selected].astype(np.int64)
        images = metadata["image_id"][selected].astype(str)
        trial_ids = metadata["trial_id"][selected].astype(str)
        source_indices = metadata["source_index"][selected].astype(np.int64)
        source_split = metadata["split"][selected].astype(str)
        source_kind = "prepared_npy"
    else:
        archive = load_part(cfg.get("archive", "EEG-ImageNet_1.pth"))
        chosen = [(index, row) for index, row in enumerate(archive["dataset"])
                  if int(row["subject"]) == subject_id]
        if len(chosen) != 4000:
            raise AssertionError(f"Expected 4000 subject rows, got {len(chosen)}")
        label_lookup = {label: index for index, label in enumerate(archive["labels"])}
        x = torch.empty((4000, 62, crop_end - crop_start), dtype=torch.float32)
        labels = np.empty(4000, dtype=np.int64)
        images = np.empty(4000, dtype="U128")
        trial_ids = np.empty(4000, dtype="U64")
        source_indices = np.empty(4000, dtype=np.int64)
        source_split = np.empty(4000, dtype="U5")
        positions = {label: 0 for label in range(80)}
        for local, (source_index, row) in enumerate(chosen):
            label = label_lookup[row["label"]]
            position = positions[label]
            positions[label] += 1
            x[local] = row["eeg_data"][:, crop_start:crop_end].float()
            labels[local] = label
            images[local] = str(row["image"])
            trial_ids[local] = f"subject-{subject_id:02d}-{source_index:06d}"
            source_indices[local] = source_index
            source_split[local] = "train" if position < 30 else "test"
        source_kind = "pth_archive"
    counts = np.bincount(labels, minlength=80)
    if tuple(x.shape) != (4000, 62, crop_end - crop_start) or not np.all(counts == 50):
        raise AssertionError("Unexpected subject data shape or class counts")
    official_train = np.flatnonzero(source_split == "train")
    test = np.flatnonzero(source_split == "test")
    fit: list[int] = []
    validation: list[int] = []
    prefix = cfg.get("development_split", {}).get("hash_prefix", "eeg-development:")
    for label in range(80):
        candidates = [int(index) for index in official_train if labels[index] == label]
        candidates.sort(key=lambda index: hashlib.sha256((prefix + images[index]).encode()).hexdigest())
        validation.extend(candidates[:3])
        fit.extend(candidates[3:])
    return {
        "waveform": x,
        "labels": torch.from_numpy(labels),
        "images": images,
        "trial_ids": trial_ids,
        "source_indices": source_indices,
        "source_kind": source_kind,
        "fit": torch.tensor(fit, dtype=torch.long),
        "validation": torch.tensor(validation, dtype=torch.long),
        "official_train": torch.from_numpy(official_train.astype(np.int64)),
        "test": torch.from_numpy(test.astype(np.int64)),
    }


def _tensor_hash(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def split_evidence(data: dict[str, Any]) -> dict[str, Any]:
    """Return auditable split counts and stable index hashes."""
    labels = data["labels"]

    def counts(indices: torch.Tensor) -> list[int]:
        return torch.bincount(labels[indices], minlength=80).tolist()

    train_images = {str(data["images"][index]) for index in data["official_train"].tolist()}
    test_images = {str(data["images"][index]) for index in data["test"].tolist()}
    return {
        "source_kind": data["source_kind"],
        "total": len(data["waveform"]),
        "fit": len(data["fit"]),
        "validation": len(data["validation"]),
        "official_train": len(data["official_train"]),
        "test": len(data["test"]),
        "per_class": {
            "fit": counts(data["fit"]),
            "validation": counts(data["validation"]),
            "official_train": counts(data["official_train"]),
            "test": counts(data["test"]),
        },
        "official_train_test_image_overlap": len(train_images & test_images),
        "fit_hash": _tensor_hash(data["fit"]),
        "validation_hash": _tensor_hash(data["validation"]),
        "official_train_hash": _tensor_hash(data["official_train"]),
        "test_hash": _tensor_hash(data["test"]),
    }
