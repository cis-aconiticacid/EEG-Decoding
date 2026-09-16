"""Run D089 masked waveform reconstruction and image-embedding retrieval.

The code enforces the D089 contract: no JEPA/EMA encoder and no category
classification head.  Development uses 27/3 rows from the official first 30;
final refits use all first 30 and touch the last 20 only once at the endpoint.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from d089_masked_retrieval import (  # noqa: E402
    CHANNELS,
    IMAGE_DIM,
    IMAGE_TOKENS,
    PATCHES,
    PATCH_SAMPLES,
    ROUTES,
    D089Encoder,
    ImageLatentPredictor,
    MaskedWaveformReconstructor,
    image_latent_loss,
    masked_waveform,
    normalized_flat,
    parameter_counts,
    periodic_hann_log_psd,
)
from eegdecoding.data import load_part  # noqa: E402


CONFIG = ROOT / "config/d089_corrected_masked_retrieval.json"


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_hash(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def module_state_hash(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.detach().cpu().contiguous()).tobytes())
    return digest.hexdigest()


def resolve_target_dir(cfg: dict[str, Any]) -> Path:
    for candidate in cfg["target_dir_candidates"]:
        path = Path(candidate)
        if not path.is_absolute():
            path = ROOT / path
        if (path / "targets.json").exists():
            return path
    raise FileNotFoundError(f"No image target directory found: {cfg['target_dir_candidates']}")


def load_subject_data(cfg: dict[str, Any]) -> dict[str, Any]:
    waveform_path = ROOT / cfg["waveform"]
    metadata_path = ROOT / cfg["metadata"]
    if waveform_path.exists() and metadata_path.exists():
        waveform = np.load(waveform_path, mmap_mode="r", allow_pickle=False)
        metadata = np.load(metadata_path, allow_pickle=False)
        selected = np.flatnonzero(metadata["subject"] == cfg["subject"])
        x = torch.from_numpy(np.asarray(waveform[selected], dtype=np.float32).copy())
        labels = metadata["label_index"][selected].astype(np.int64)
        images = metadata["image_id"][selected].astype(str)
        trial_ids = metadata["trial_id"][selected].astype(str)
        source_split = metadata["split"][selected].astype(str)
        source_indices = metadata["source_index"][selected].astype(np.int64)
        source_kind = "prepared_npy"
    else:
        archive_path = ROOT / cfg["archive"]
        archive = load_part(archive_path)
        chosen = [(index, row) for index, row in enumerate(archive["dataset"])
                  if int(row["subject"]) == cfg["subject"]]
        if len(chosen) != 4000:
            raise AssertionError(f"Expected 4000 subject rows, got {len(chosen)}")
        label_lookup = {label: index for index, label in enumerate(archive["labels"])}
        x = torch.empty((4000, CHANNELS, PATCHES * PATCH_SAMPLES), dtype=torch.float32)
        labels = np.empty(4000, dtype=np.int64)
        images = np.empty(4000, dtype="U128")
        trial_ids = np.empty(4000, dtype="U64")
        source_indices = np.empty(4000, dtype=np.int64)
        source_split = np.empty(4000, dtype="U5")
        positions: dict[int, int] = {label: 0 for label in range(80)}
        for local, (source_index, row) in enumerate(chosen):
            label = label_lookup[row["label"]]
            position = positions[label]
            positions[label] += 1
            x[local] = row["eeg_data"][:, cfg["crop"][0]:cfg["crop"][1]].float()
            labels[local] = label
            images[local] = str(row["image"])
            trial_ids[local] = f"d089-s{cfg['subject']:02d}-{source_index:06d}"
            source_indices[local] = source_index
            source_split[local] = "train" if position < 30 else "test"
        del archive
        source_kind = "pth_archive"

    if tuple(x.shape) != (4000, CHANNELS, PATCHES * PATCH_SAMPLES):
        raise AssertionError(f"Unexpected EEG shape {tuple(x.shape)}")
    if not torch.isfinite(x).all():
        raise AssertionError("EEG waveform contains non-finite values")
    counts = np.bincount(labels, minlength=80)
    if not np.all(counts == 50):
        raise AssertionError(f"Expected 50 rows per class: {counts.tolist()}")
    official_train = np.flatnonzero(source_split == "train")
    test = np.flatnonzero(source_split == "test")
    if (len(official_train), len(test)) != (2400, 1600):
        raise AssertionError((len(official_train), len(test)))
    fit, validation = [], []
    for label in range(80):
        candidates = [int(index) for index in official_train if labels[index] == label]
        candidates.sort(key=lambda index: hashlib.sha256(
            (cfg["development_split"]["hash_prefix"] + images[index]).encode()
        ).hexdigest())
        validation.extend(candidates[:3])
        fit.extend(candidates[3:])
    fit = np.asarray(fit, dtype=np.int64)
    validation = np.asarray(validation, dtype=np.int64)
    if (len(fit), len(validation)) != (2160, 240):
        raise AssertionError((len(fit), len(validation)))
    if set(images[official_train]) & set(images[test]):
        raise AssertionError("Official train/test image IDs overlap")
    return {
        "waveform": x,
        "labels": torch.from_numpy(labels),
        "images": images,
        "trial_ids": trial_ids,
        "source_indices": source_indices,
        "source_kind": source_kind,
        "fit": torch.from_numpy(fit),
        "validation": torch.from_numpy(validation),
        "official_train": torch.from_numpy(official_train.astype(np.int64)),
        "test": torch.from_numpy(test.astype(np.int64)),
    }


def fit_statistics(waveform: torch.Tensor, fit: torch.Tensor, device: torch.device) -> dict[str, torch.Tensor]:
    total = torch.zeros(CHANNELS, dtype=torch.float64)
    square = torch.zeros(CHANNELS, dtype=torch.float64)
    count = 0
    for indices in fit.split(64):
        value = waveform[indices].double()
        total += value.sum((0, 2))
        square += value.square().sum((0, 2))
        count += value.shape[0] * value.shape[2]
    mean = total / count
    scale = (square / count - mean.square()).clamp_min(1e-12).sqrt()
    standardized = (waveform - mean.float()[None, :, None]) / scale.float()[None, :, None]
    standardized = standardized.to(device)
    frequency_values = []
    for indices in fit.split(64):
        frequency_values.append(periodic_hann_log_psd(standardized[indices.to(device)]).cpu())
    frequency = torch.cat(frequency_values)
    frequency_mean = frequency.mean(0)
    frequency_scale = frequency.std(0, correction=0).clamp_min(1e-6)
    return {
        "wave_mean": mean.float(),
        "wave_scale": scale.float(),
        "waveform": standardized,
        "frequency_mean": frequency_mean.to(device),
        "frequency_scale": frequency_scale.to(device),
    }


def load_image_targets(cfg: dict[str, Any], data: dict[str, Any], device: torch.device) -> dict[str, Any]:
    target_dir = resolve_target_dir(cfg)
    contract_path = target_dir / "targets.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract["shape_per_image"] != [IMAGE_TOKENS, IMAGE_DIM] or contract["all_image_count"] != 4000:
        raise AssertionError("Image target shape/count differs from D089 contract")
    targets = torch.empty((4000, IMAGE_TOKENS, IMAGE_DIM), dtype=torch.float32, device=device)
    records = []
    for split in ("train", "monitor_val", "final_holdout"):
        meta = contract["splits"][split]
        path = target_dir / f"{split}.npy"
        if not meta["complete"] or sha256(path) != meta["array_sha256"]:
            raise AssertionError(f"Image target hash differs: {split}")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if list(array.shape) != meta["shape"] or array.dtype != np.float32:
            raise AssertionError(f"Image target array differs: {split}")
        offset = len(records)
        for start in range(0, len(array), 32):
            stop = min(start + 32, len(array))
            chunk = np.asarray(array[start:stop]).copy()
            if not np.isfinite(chunk).all():
                raise AssertionError(f"Image target contains non-finite values: {split}[{start}:{stop}]")
            targets[offset + start:offset + stop].copy_(torch.from_numpy(chunk))
        records.extend(meta["images"])
    lookup = {record["image_id"]: index for index, record in enumerate(records)}
    if len(lookup) != 4000:
        raise AssertionError("Image embedding IDs are not unique")
    image_indices = torch.tensor([lookup[image] for image in data["images"]], dtype=torch.long, device=device)
    gallery_labels = torch.tensor([int(record["label_index"]) for record in records], device=device)
    if not torch.equal(gallery_labels[image_indices].cpu(), data["labels"]):
        raise AssertionError("EEG image IDs and embedding labels disagree")
    gallery = torch.empty((4000, IMAGE_TOKENS * IMAGE_DIM), dtype=torch.float32, device=device)
    for start in range(0, 4000, 32):
        gallery[start:start + 32] = normalized_flat(targets[start:start + 32])
    candidates = {
        "all": torch.arange(4000, device=device),
        "coarse": torch.flatnonzero(gallery_labels >= 40),
        **{f"fine{group}": torch.flatnonzero(
            (gallery_labels >= 8 * group) & (gallery_labels < 8 * group + 8)
        ) for group in range(5)},
    }
    expected = {"all": 4000, "coarse": 2000, **{f"fine{group}": 400 for group in range(5)}}
    if {key: len(value) for key, value in candidates.items()} != expected:
        raise AssertionError("Embedding candidate counts differ")
    return {
        "targets": targets,
        "gallery": gallery,
        "gallery_labels": gallery_labels,
        "image_indices": image_indices,
        "records": records,
        "candidates": candidates,
        "target_dir": str(target_dir),
        "contract_sha256": sha256(contract_path),
    }


def make_masks(size: int, ratio: float, generator: torch.Generator, device: torch.device) -> torch.Tensor:
    exact = min(4.0, max(1.0, PATCHES * ratio))
    lower = int(math.floor(exact))
    fraction = exact - lower
    masks = torch.zeros((size, PATCHES), dtype=torch.bool)
    for row in range(size):
        length = lower + int(torch.rand((), generator=generator).item() < fraction)
        start = int(torch.randint(PATCHES - length + 1, (), generator=generator).item())
        masks[row, start:start + length] = True
    return masks.to(device)


def cosine_lr(step: int, total: int, warmup: int, peak: float, minimum: float) -> float:
    if step <= warmup:
        return peak * step / max(1, warmup)
    phase = (step - warmup) / max(1, total - warmup)
    return minimum + 0.5 * (peak - minimum) * (1 + math.cos(math.pi * phase))


def shuffled_cycle(indices: torch.Tensor, count: int, rng: np.random.Generator, state: dict[str, Any]) -> torch.Tensor:
    if "order" not in state:
        state["order"] = indices.cpu().numpy().copy()
        rng.shuffle(state["order"])
        state["cursor"] = 0
    selected = []
    while len(selected) < count:
        remaining = len(state["order"]) - state["cursor"]
        take = min(count - len(selected), remaining)
        selected.extend(state["order"][state["cursor"]:state["cursor"] + take].tolist())
        state["cursor"] += take
        if state["cursor"] == len(state["order"]):
            rng.shuffle(state["order"])
            state["cursor"] = 0
    return torch.tensor(selected, dtype=torch.long, device=indices.device)


def save_checkpoint(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def checkpoint_contract(route: str, stage: str, fit: torch.Tensor) -> dict[str, Any]:
    return {
        "task": "D089", "route": route, "stage": stage,
        "config_sha256": sha256(CONFIG),
        "model_sha256": sha256(ROOT / "src/d089_masked_retrieval.py"),
        "runner_sha256": sha256(Path(__file__)),
        "fit_hash": tensor_hash(fit),
    }


def pretrain(
    route: str,
    stage: str,
    waveform: torch.Tensor,
    fit: torch.Tensor,
    frequency_mean: torch.Tensor,
    frequency_scale: torch.Tensor,
    cfg: dict[str, Any],
    run: Path,
) -> Path:
    settings = cfg["pretrain"]
    device = waveform.device
    run.mkdir(parents=True, exist_ok=True)
    contract = checkpoint_contract(route, stage, fit)
    final_path = run / f"pretrain_step{settings['steps']:04d}.pt"
    if final_path.exists():
        completed = torch.load(final_path, map_location="cpu", weights_only=False)
        if completed.get("contract") != contract or completed.get("step") != settings["steps"]:
            raise AssertionError("Existing pretraining checkpoint contract differs")
        return final_path
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])
    model = MaskedWaveformReconstructor(D089Encoder(route, activation_checkpointing=False)).to(device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=settings["peak_lr"], weight_decay=settings["weight_decay"],
    )
    rng = np.random.default_rng(cfg["seed"] + 8900)
    mask_rng = torch.Generator().manual_seed(cfg["seed"] + 8901)
    cycle: dict[str, Any] = {}
    log = run / "pretrain.jsonl"
    total_steps = settings["steps"]
    warmup = max(1, int(total_steps * settings["warmup_fraction"]))
    started = time.perf_counter()
    fit_device = fit.to(device)
    start_step = 0
    recovery_path = run / "pretrain_recovery.pt"
    if recovery_path.exists():
        recovered = torch.load(recovery_path, map_location=device, weights_only=False)
        if recovered.get("contract") != contract or recovered.get("kind") != "masked_waveform_reconstruction":
            raise AssertionError("Pretraining recovery checkpoint contract differs")
        model.load_state_dict(recovered["model"], strict=True)
        optimizer.load_state_dict(recovered["optimizer"])
        rng.bit_generator.state = recovered["rng"]
        mask_rng.set_state(recovered["mask_rng"].cpu())
        cycle = recovered["cycle"]
        torch.set_rng_state(recovered["torch_rng"].cpu())
        torch.cuda.set_rng_state_all([value.cpu() for value in recovered["cuda_rng"]])
        start_step = int(recovered["step"])
    for step in range(start_step + 1, total_steps + 1):
        logical = shuffled_cycle(fit_device, settings["logical_batch"], rng, cycle)
        progress = min(1.0, step / max(1, int(total_steps * settings["mask_schedule_fraction"])))
        ratio = settings["mask_start"] + (settings["mask_end"] - settings["mask_start"]) * progress
        hidden = make_masks(len(logical), ratio, mask_rng, device)
        lr = cosine_lr(step, total_steps, warmup, settings["peak_lr"], settings["min_lr"])
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        totals = {"loss": 0.0, "wave": 0.0, "slope": 0.0}
        for start in range(0, len(logical), settings["microbatch"]):
            ids = logical[start:start + settings["microbatch"]]
            mask = hidden[start:start + settings["microbatch"]]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, parts = model.loss(waveform[ids], mask, frequency_mean, frequency_scale)
            weight = len(ids) / len(logical)
            (loss * weight).backward()
            totals["loss"] += float(loss.detach()) * len(ids)
            totals["wave"] += float(parts["wave"]) * len(ids)
            totals["slope"] += float(parts["slope"]) * len(ids)
        grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip"]))
        optimizer.step()
        if step == 1 or step % 50 == 0:
            append_jsonl(log, {"task": "D089", "stage": stage, "route": route, "step": step,
                               "loss": totals["loss"] / len(logical), "wave": totals["wave"] / len(logical),
                               "slope": totals["slope"] / len(logical), "mask_ratio": ratio, "lr": lr,
                               "grad_norm": grad, "seconds": time.perf_counter() - started,
                               "allocated_gib": torch.cuda.memory_allocated() / 2**30})
        if step % 100 == 0 or step in settings["checkpoint_steps"]:
            state = {
                "task": "D089", "kind": "masked_waveform_reconstruction", "jepa": False,
                "stage": stage, "route": route, "step": step, "contract": contract,
                "model": model.state_dict(), "encoder": model.encoder.state_dict(),
                "reconstruction_head": model.head.state_dict(), "optimizer": optimizer.state_dict(),
                "rng": rng.bit_generator.state, "mask_rng": mask_rng.get_state(),
                "cycle": cycle, "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all(),
            }
            save_checkpoint(recovery_path, state)
            if step in settings["checkpoint_steps"]:
                save_checkpoint(run / f"pretrain_step{step:04d}.pt", state)
    del model, optimizer
    torch.cuda.empty_cache()
    return final_path


def f1_macro(actual: np.ndarray, predicted: np.ndarray, labels: list[int]) -> float:
    values = []
    for label in labels:
        tp = int(np.sum((actual == label) & (predicted == label)))
        fp = int(np.sum((actual != label) & (predicted == label)))
        fn = int(np.sum((actual == label) & (predicted != label)))
        denominator = 2 * tp + fp + fn
        values.append(0.0 if denominator == 0 else 2 * tp / denominator)
    return float(np.mean(values))


def category_topk(
    scores: torch.Tensor,
    candidates: torch.Tensor,
    gallery_labels: torch.Tensor,
    first_label: int,
    class_count: int,
    k: int = 5,
) -> torch.Tensor:
    """Rank classes by the best matching image in each class.

    This keeps image retrieval as the primitive operation.  It does not create
    class prototypes or a learned category head: a class score is simply the
    maximum cosine similarity among that class's candidate images.
    """
    candidate_scores = scores[:, candidates]
    local_labels = gallery_labels[candidates] - first_label
    if int(local_labels.min()) != 0 or int(local_labels.max()) != class_count - 1:
        raise AssertionError("Candidate gallery labels do not match the requested category range")
    class_scores = torch.full(
        (len(scores), class_count), -torch.inf, dtype=scores.dtype, device=scores.device
    )
    class_scores.scatter_reduce_(
        1,
        local_labels[None].expand(len(scores), -1),
        candidate_scores,
        reduce="amax",
        include_self=True,
    )
    return class_scores.topk(min(k, class_count), dim=1).indices + first_label


@torch.inference_mode()
def evaluate_retrieval(
    model: ImageLatentPredictor,
    waveform: torch.Tensor,
    indices: torch.Tensor,
    data: dict[str, Any],
    images: dict[str, Any],
    frequency_mean: torch.Tensor,
    frequency_scale: torch.Tensor,
    microbatch: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    actual_labels, all_predictions = [], []
    coarse_actual, coarse_predictions = [], []
    fine_actual = [[] for _ in range(5)]
    fine_predictions = [[] for _ in range(5)]
    mse_sum = 0.0
    image_top1 = image_top5 = 0
    all_category_top5 = coarse_category_top5 = 0
    fine_category_top5 = [0 for _ in range(5)]
    target_similarity_sum = target_margin_sum = winner_margin_sum = 0.0
    predictions = []
    labels_cpu = data["labels"].numpy()
    for batch_indices in indices.split(microbatch):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            predicted = model(waveform[batch_indices], frequency_mean, frequency_scale)
        actual_image = images["image_indices"][batch_indices]
        target = images["targets"][actual_image]
        per_mse = (predicted.float() - target).square().flatten(1).mean(1)
        mse_sum += float(per_mse.sum())
        query = normalized_flat(predicted)
        scores = []
        for gallery_chunk in images["gallery"].split(256):
            scores.append(query @ gallery_chunk.T)
        scores = torch.cat(scores, dim=1)
        all_top5 = scores.topk(5, dim=1).indices
        all_winner = all_top5[:, 0]
        all_class_top5 = category_topk(
            scores, images["candidates"]["all"], images["gallery_labels"], 0, 80
        )
        predicted_labels = all_class_top5[:, 0]
        image_winner_labels = images["gallery_labels"][all_winner]
        if not torch.equal(predicted_labels, image_winner_labels):
            raise AssertionError("Nearest-image and max-image-per-class predictions disagree")
        true_labels = data["labels"][batch_indices]
        actual_labels.extend(true_labels.cpu().tolist())
        all_predictions.extend(predicted_labels.cpu().tolist())
        image_top1 += int((all_winner == actual_image).sum())
        image_top5 += int((all_top5 == actual_image[:, None]).any(1).sum())
        all_category_top5 += int((all_class_top5 == true_labels[:, None]).any(1).sum())

        row_index = torch.arange(len(scores), device=scores.device)
        target_similarity = scores[row_index, actual_image]
        without_target = scores.clone()
        without_target[row_index, actual_image] = -torch.inf
        best_other = without_target.amax(1)
        winner_two = scores.topk(2, dim=1).values
        target_similarity_sum += float(target_similarity.sum())
        target_margin_sum += float((target_similarity - best_other).sum())
        winner_margin_sum += float((winner_two[:, 0] - winner_two[:, 1]).sum())

        for task, candidates in images["candidates"].items():
            if task == "all":
                continue
            if task == "coarse":
                sample = true_labels >= 40
                group = None
            else:
                group = int(task[-1])
                sample = (true_labels >= 8 * group) & (true_labels < 8 * group + 8)
            if not sample.any():
                continue
            winners = candidates[scores[sample][:, candidates].argmax(1)]
            found_labels = images["gallery_labels"][winners].cpu().tolist()
            source_labels = true_labels[sample].cpu().tolist()
            if task == "coarse":
                category_top5 = category_topk(
                    scores[sample], candidates, images["gallery_labels"], 40, 40
                )
                coarse_category_top5 += int(
                    (category_top5 == true_labels[sample, None]).any(1).sum()
                )
                coarse_actual.extend(source_labels)
                coarse_predictions.extend(found_labels)
            else:
                category_top5 = category_topk(
                    scores[sample], candidates, images["gallery_labels"], 8 * group, 8
                )
                fine_category_top5[group] += int(
                    (category_top5 == true_labels[sample, None]).any(1).sum()
                )
                fine_actual[group].extend(source_labels)
                fine_predictions[group].extend(found_labels)

        for local, source_index in enumerate(batch_indices.cpu().tolist()):
            winner = int(all_winner[local])
            predictions.append({
                "trial_id": str(data["trial_ids"][source_index]),
                "actual_image": str(data["images"][source_index]),
                "actual_label": int(labels_cpu[source_index]),
                "predicted_image": str(images["records"][winner]["image_id"]),
                "predicted_label": int(images["gallery_labels"][winner]),
                "similarity": float(scores[local, winner]),
                "actual_image_similarity": float(target_similarity[local]),
                "actual_image_margin": float(target_similarity[local] - best_other[local]),
                "winner_margin": float(winner_two[local, 0] - winner_two[local, 1]),
                "top5_category_labels": [int(value) for value in all_class_top5[local].cpu().tolist()],
                "latent_mse": float(per_mse[local]),
            })

    actual = np.asarray(actual_labels)
    all_pred = np.asarray(all_predictions)
    coarse_y = np.asarray(coarse_actual)
    coarse_p = np.asarray(coarse_predictions)
    fine_groups = []
    for group in range(5):
        y = np.asarray(fine_actual[group])
        p = np.asarray(fine_predictions[group])
        fine_groups.append({
            "group": group, "n": len(y), "accuracy": float(np.mean(y == p)),
            "top5_accuracy": fine_category_top5[group] / len(y),
            "macro_f1": f1_macro(y, p, list(range(8 * group, 8 * group + 8))),
        })
    result = {
        "n": len(actual),
        "latent_mse": mse_sum / len(actual),
        "all": {"accuracy": float(np.mean(actual == all_pred)), "top5_accuracy": all_category_top5 / len(actual),
                "macro_f1": f1_macro(actual, all_pred, list(range(80))), "gallery_size": 4000},
        "coarse": {"n": len(coarse_y), "accuracy": float(np.mean(coarse_y == coarse_p)),
                   "top5_accuracy": coarse_category_top5 / len(coarse_y),
                   "macro_f1": f1_macro(coarse_y, coarse_p, list(range(40, 80))), "gallery_size": 2000},
        "fine": {"accuracy": float(np.mean([group["accuracy"] for group in fine_groups])),
                 "top5_accuracy": float(np.mean([group["top5_accuracy"] for group in fine_groups])),
                 "macro_f1": float(np.mean([group["macro_f1"] for group in fine_groups])),
                 "groups": fine_groups, "gallery_size_per_group": 400},
        "exact_image": {"top1": image_top1 / len(actual), "top5": image_top5 / len(actual),
                         "gallery_size": 4000},
        "similarity": {
            "mean_actual_image": target_similarity_sum / len(actual),
            "mean_actual_vs_best_other_margin": target_margin_sum / len(actual),
            "mean_top1_vs_top2_margin": winner_margin_sum / len(actual),
        },
        "random_chance": {
            "all": {"top1": 1 / 80, "top5": 5 / 80},
            "coarse": {"top1": 1 / 40, "top5": 5 / 40},
            "fine": {"top1": 1 / 8, "top5": 5 / 8},
            "exact_image_all": {"top1": 1 / 4000, "top5": 5 / 4000},
        },
    }
    return result, predictions


def selection_key(row: dict[str, Any]) -> tuple[float, float, float, int]:
    metric = row["validation"]
    mean_granularity = np.mean([metric["all"]["accuracy"], metric["coarse"]["accuracy"], metric["fine"]["accuracy"]])
    return metric["all"]["accuracy"], float(mean_granularity), -metric["latent_mse"], -row["epoch"]


def finetune(
    route: str,
    stage: str,
    pretrain_path: Path,
    waveform: torch.Tensor,
    fit: torch.Tensor,
    evaluation_indices: torch.Tensor | None,
    data: dict[str, Any],
    images: dict[str, Any],
    frequency_mean: torch.Tensor,
    frequency_scale: torch.Tensor,
    cfg: dict[str, Any],
    run: Path,
    epochs: int,
) -> dict[str, Any]:
    settings = cfg["finetune"]
    device = waveform.device
    run.mkdir(parents=True, exist_ok=True)
    result_path = run / "result.json"
    if result_path.exists():
        completed = json.loads(result_path.read_text(encoding="utf-8"))
        if completed.get("contract") != checkpoint_contract(route, stage, fit):
            raise AssertionError("Existing fine-tuning result contract differs")
        return completed
    contract = checkpoint_contract(route, stage, fit)
    torch.manual_seed(cfg["seed"] + 1)
    torch.cuda.manual_seed_all(cfg["seed"] + 1)
    encoder = D089Encoder(route, activation_checkpointing=False)
    pretrain_state = torch.load(pretrain_path, map_location="cpu", weights_only=False)
    if pretrain_state.get("jepa") is not False or pretrain_state.get("kind") != "masked_waveform_reconstruction":
        raise AssertionError("Pretraining checkpoint is not D089 waveform reconstruction")
    encoder.load_state_dict(pretrain_state["encoder"], strict=True)
    model = ImageLatentPredictor(encoder).to(device)
    encoder_parameters = [parameter for parameter in model.encoder.parameters() if parameter.requires_grad]
    head_parameters = [parameter for name, parameter in model.named_parameters()
                       if not name.startswith("encoder.") and parameter.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": encoder_parameters, "lr": settings["head_peak_lr"] * settings["encoder_lr_multiplier"]},
        {"params": head_parameters, "lr": settings["head_peak_lr"]},
    ], weight_decay=settings["weight_decay"])
    rng = np.random.default_rng(cfg["seed"] + 8902)
    history = []
    started = time.perf_counter()
    fit_device = fit.to(device)
    total_updates = epochs * math.ceil(len(fit) / settings["logical_batch"])
    warmup_updates = settings["warmup_epochs"] * math.ceil(len(fit) / settings["logical_batch"])
    update = 0
    start_epoch = 0
    recovery_path = run / "finetune_recovery.pt"
    if recovery_path.exists():
        recovered = torch.load(recovery_path, map_location=device, weights_only=False)
        if recovered.get("contract") != contract or recovered.get("kind") != "image_latent_retrieval":
            raise AssertionError("Fine-tuning recovery checkpoint contract differs")
        model.load_state_dict(recovered["model"], strict=True)
        optimizer.load_state_dict(recovered["optimizer"])
        history = recovered["history"]
        start_epoch = int(recovered["epoch"])
        update = int(recovered["update"])
        rng.bit_generator.state = recovered["rng"]
        torch.set_rng_state(recovered["torch_rng"].cpu())
        torch.cuda.set_rng_state_all([value.cpu() for value in recovered["cuda_rng"]])
    for epoch in range(start_epoch + 1, epochs + 1):
        order = fit.cpu().numpy().copy()
        rng.shuffle(order)
        order = torch.from_numpy(order).to(device)
        model.train()
        totals = {"loss": 0.0, "mse": 0.0, "cosine_loss": 0.0, "contrastive": 0.0}
        seen = 0
        for logical in order.split(settings["logical_batch"]):
            update += 1
            head_lr = cosine_lr(update, total_updates, warmup_updates,
                                settings["head_peak_lr"], settings["head_min_lr"])
            optimizer.param_groups[0]["lr"] = head_lr * settings["encoder_lr_multiplier"]
            optimizer.param_groups[1]["lr"] = head_lr
            optimizer.zero_grad(set_to_none=True)
            for ids in logical.split(settings["microbatch"]):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    predicted = model(waveform[ids], frequency_mean, frequency_scale)
                    target = images["targets"][images["image_indices"][ids]]
                    loss, parts = image_latent_loss(predicted, target, settings["temperature"])
                weight = len(ids) / len(logical)
                (loss * weight).backward()
                for key in totals:
                    value = loss if key == "loss" else parts[key]
                    totals[key] += float(value.detach()) * len(ids)
                seen += len(ids)
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip"])
            optimizer.step()
        row: dict[str, Any] = {
            "task": "D089", "stage": stage, "route": route, "epoch": epoch,
            **{key: value / seen for key, value in totals.items()},
            "head_lr": optimizer.param_groups[1]["lr"], "encoder_lr": optimizer.param_groups[0]["lr"],
            "seconds": time.perf_counter() - started,
            "allocated_gib": torch.cuda.memory_allocated() / 2**30,
        }
        if evaluation_indices is not None and epoch in settings["evaluation_epochs"]:
            validation, _ = evaluate_retrieval(model, waveform, evaluation_indices.to(device), data, images,
                                               frequency_mean, frequency_scale, settings["microbatch"])
            row["validation"] = validation
        history.append(row)
        append_jsonl(run / "finetune.jsonl", row)
        if epoch in settings["checkpoint_epochs"] or epoch == epochs:
            state = {
                "task": "D089", "kind": "image_latent_retrieval", "classification_head": False,
                "jepa": False, "stage": stage, "route": route, "epoch": epoch,
                "update": update, "contract": contract, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "history": history,
                "rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all(),
            }
            save_checkpoint(recovery_path, state)
            save_checkpoint(run / f"finetune_epoch{epoch:03d}.pt", state)
    result: dict[str, Any] = {"history": history, "parameters": parameter_counts(model), "contract": contract}
    if evaluation_indices is not None:
        candidates = [row for row in history if "validation" in row]
        best = max(candidates, key=selection_key)
        result["selected_epoch"] = best["epoch"]
        result["selected_validation"] = best["validation"]
    else:
        test, predictions = evaluate_retrieval(model, waveform, data["test"].to(device), data, images,
                                               frequency_mean, frequency_scale, settings["microbatch"])
        result["test"] = test
        atomic_json(run / "test_predictions.json", predictions)
    atomic_json(result_path, result)
    del model, optimizer
    torch.cuda.empty_cache()
    return result


def split_evidence(data: dict[str, Any]) -> dict[str, Any]:
    labels = data["labels"]
    def counts(indices: torch.Tensor) -> list[int]:
        return torch.bincount(labels[indices], minlength=80).tolist()

    train_images = {str(data["images"][index]) for index in data["official_train"].tolist()}
    test_images = {str(data["images"][index]) for index in data["test"].tolist()}
    return {
        "source_kind": data["source_kind"],
        "total": len(data["waveform"]), "fit": len(data["fit"]), "validation": len(data["validation"]),
        "official_train": len(data["official_train"]), "test": len(data["test"]),
        "per_class": {
            "total": counts(torch.arange(len(labels))), "fit": counts(data["fit"]),
            "validation": counts(data["validation"]), "official_train": counts(data["official_train"]),
            "test": counts(data["test"]),
        },
        "unique_image_ids": len(set(str(value) for value in data["images"])),
        "official_train_test_image_overlap": len(train_images & test_images),
        "fit_hash": tensor_hash(data["fit"]), "validation_hash": tensor_hash(data["validation"]),
        "official_train_hash": tensor_hash(data["official_train"]), "test_hash": tensor_hash(data["test"]),
    }


def smoke(device: torch.device) -> dict[str, Any]:
    waveform = torch.randn(1, CHANNELS, PATCHES * PATCH_SAMPLES, device=device)
    hidden = torch.tensor([[False, True, True, False, False, False, False, False]], device=device)
    frequency_mean = torch.zeros(CHANNELS, 32, device=device)
    frequency_scale = torch.ones(CHANNELS, 32, device=device)
    output = {}
    initialization_hashes = {}
    for route in ROUTES:
        torch.manual_seed(17)
        reconstructor = MaskedWaveformReconstructor(D089Encoder(route)).to(device)
        changed = waveform.clone().reshape(1, CHANNELS, PATCHES, PATCH_SAMPLES)
        changed[:, :, hidden[0]] = torch.randn_like(changed[:, :, hidden[0]]) * 100
        changed = changed.reshape_as(waveform)
        if not torch.equal(masked_waveform(waveform, hidden), masked_waveform(changed, hidden)):
            raise AssertionError("Masked waveform still contains hidden target samples")
        loss, _ = reconstructor.loss(waveform, hidden, frequency_mean, frequency_scale)
        loss.backward()
        if reconstructor.encoder.wave_patch[0].weight.grad is None:
            raise AssertionError("Reconstruction gradient did not enter EEG encoder")
        reconstructor.eval()
        with torch.no_grad():
            original_output = reconstructor(waveform, hidden, frequency_mean, frequency_scale)
            changed_output = reconstructor(changed, hidden, frequency_mean, frequency_scale)
        leakage = float((original_output - changed_output).abs().max())
        if leakage != 0.0:
            raise AssertionError(f"Masked target leaked into reconstruction: {leakage}")
        encoder = copy.deepcopy(reconstructor.encoder)
        del reconstructor
        image_model = ImageLatentPredictor(encoder).to(device)
        initialization_hashes[route] = module_state_hash(image_model)
        prediction = image_model(waveform, frequency_mean, frequency_scale)
        if tuple(prediction.shape) != (1, IMAGE_TOKENS, IMAGE_DIM):
            raise AssertionError(tuple(prediction.shape))
        target = torch.randn_like(prediction)
        image_loss, _ = image_latent_loss(prediction, target)
        image_loss.backward()
        if image_model.encoder.wave_patch[0].weight.grad is None or image_model.output[-1].weight.grad is None:
            raise AssertionError("Image loss gradient path is incomplete")
        names = list(dict(image_model.named_modules()))
        if any("classifier" in name.lower() or "jepa" in name.lower() for name in names):
            raise AssertionError("Forbidden classifier/JEPA module present")
        if any(isinstance(module, torch.nn.Linear) and module.out_features == 80 for module in image_model.modules()):
            raise AssertionError("Forbidden 80-class linear head present")
        toy_gallery = torch.eye(4, device=device)
        toy_query = torch.tensor([[0.0, 0.9, 0.1, 0.0]], device=device)
        toy_winner = int((toy_query @ toy_gallery.T).argmax(1))
        toy_labels = torch.tensor([0, 1, 0, 1], device=device)
        if toy_winner != 1 or int(toy_labels[toy_winner]) != 1:
            raise AssertionError("Toy embedding retrieval failed")
        toy_class_top3 = category_topk(
            toy_query @ toy_gallery.T, torch.arange(4, device=device), toy_labels, 0, 2, 2
        )
        if toy_class_top3.tolist() != [[1, 0]]:
            raise AssertionError("Toy category retrieval failed")
        output[route] = {"reconstruction_loss": float(loss.detach()), "image_loss": float(image_loss.detach()),
                         "image_shape": list(prediction.shape), "parameters": parameter_counts(image_model)}
        del image_model
        torch.cuda.empty_cache()
    if len(set(initialization_hashes.values())) != 1:
        raise AssertionError("Route initializations differ before routing behavior is applied")
    return {"status": "PASS", "device": str(device), "routes": output,
            "initialization_hash": next(iter(initialization_hashes.values())),
            "forbidden_modules_absent": True, "masked_input_invariance": True,
            "masked_model_output_invariance": True, "toy_embedding_retrieval": True,
            "toy_category_retrieval": True}


def formal_preflight(
    cfg: dict[str, Any], data: dict[str, Any], images: dict[str, Any], device: torch.device
) -> dict[str, Any]:
    expected_uuid = cfg["gpu_uuid"]
    inventory = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,name,memory.total", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    ).stdout.splitlines()
    parsed = [[field.strip() for field in row.split(",", 3)] for row in inventory]
    expected_rows = [row for row in parsed if row[0] == str(cfg["gpu_physical_index"]) and row[1] == expected_uuid]
    if len(expected_rows) != 1:
        raise AssertionError(f"Configured A100 physical index/UUID mismatch: {parsed}")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != expected_uuid:
        raise AssertionError("Formal D089 must run inside the assigned UUID lock")

    split = split_evidence(data)
    expected_counts = {
        "total": [50] * 80, "fit": [27] * 80, "validation": [3] * 80,
        "official_train": [30] * 80, "test": [20] * 80,
    }
    if split["per_class"] != expected_counts or split["official_train_test_image_overlap"] != 0:
        raise AssertionError("D089 data split contract failed")
    if len(images["image_indices"].unique()) != 4000:
        raise AssertionError("Subject 0 does not map one-to-one to 4,000 image embeddings")
    if tuple(images["targets"].shape) != (4000, IMAGE_TOKENS, IMAGE_DIM):
        raise AssertionError("Image target tensor shape differs")
    if tuple(images["gallery"].shape) != (4000, IMAGE_TOKENS * IMAGE_DIM):
        raise AssertionError("Image gallery tensor shape differs")
    gallery_norm_error = float((images["gallery"].norm(dim=1) - 1).abs().max())
    if gallery_norm_error > 2e-5:
        raise AssertionError(f"Image gallery normalization failed: {gallery_norm_error}")

    smoke_result = smoke(device)

    logical_batch = int(cfg["pretrain"]["logical_batch"])
    if logical_batch != 64 or int(cfg["pretrain"]["microbatch"]) != logical_batch \
            or int(cfg["finetune"]["microbatch"]) != logical_batch:
        raise AssertionError("D089 preflight expects a full 64-sample microbatch in both stages")
    probe_indices = data["fit"][:logical_batch].to(device)
    probe_waveform = data["waveform"][probe_indices.cpu()].to(device)
    probe_mean = probe_waveform.mean((0, 2), keepdim=True)
    probe_scale = probe_waveform.std((0, 2), correction=0, keepdim=True).clamp_min(1e-6)
    probe_waveform = (probe_waveform - probe_mean) / probe_scale
    probe_frequency_mean = torch.zeros(CHANNELS, 32, device=device)
    probe_frequency_scale = torch.ones(CHANNELS, 32, device=device)
    probe_hidden = make_masks(
        logical_batch, 0.5, torch.Generator().manual_seed(cfg["seed"] + 8990), device
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    probe_reconstructor = MaskedWaveformReconstructor(D089Encoder("separate_conv")).to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        probe_reconstruction_loss, _ = probe_reconstructor.loss(
            probe_waveform, probe_hidden, probe_frequency_mean, probe_frequency_scale
        )
    probe_reconstruction_loss.backward()
    torch.cuda.synchronize(device)
    reconstruction_peak = torch.cuda.max_memory_allocated(device) / 2**30
    if not torch.isfinite(probe_reconstruction_loss):
        raise AssertionError("Batch-64 reconstruction capacity probe is non-finite")
    del probe_reconstructor, probe_reconstruction_loss
    torch.cuda.empty_cache()

    torch.cuda.reset_peak_memory_stats(device)
    probe_image_model = ImageLatentPredictor(D089Encoder("separate_conv")).to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        probe_prediction = probe_image_model(probe_waveform, probe_frequency_mean, probe_frequency_scale)
        probe_target = images["targets"][images["image_indices"][probe_indices]]
        probe_image_loss, _ = image_latent_loss(
            probe_prediction, probe_target, cfg["finetune"]["temperature"]
        )
    probe_image_loss.backward()
    torch.cuda.synchronize(device)
    image_peak = torch.cuda.max_memory_allocated(device) / 2**30
    if not torch.isfinite(probe_image_loss):
        raise AssertionError("Batch-64 image capacity probe is non-finite")
    del probe_image_model, probe_prediction, probe_target, probe_image_loss, probe_waveform
    torch.cuda.empty_cache()
    source_text = (ROOT / "src/d089_masked_retrieval.py").read_text(encoding="utf-8")
    runner_text = Path(__file__).read_text(encoding="utf-8")
    forbidden_jepa_class = "D088" + "JEPA"
    forbidden_category_head = "nn.Linear(DIM, " + "80)"
    if forbidden_jepa_class in source_text + runner_text or forbidden_category_head in source_text + runner_text:
        raise AssertionError("Forbidden D088 JEPA or category head implementation found")

    return {
        "status": "PASS",
        "task": "D089",
        "gpu": {
            "physical_index": int(expected_rows[0][0]), "uuid": expected_rows[0][1],
            "name": expected_rows[0][2], "memory_total_mib": int(expected_rows[0][3]),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch_device_name": torch.cuda.get_device_name(device),
        },
        "split": split,
        "image_targets": {
            "target_dir": images["target_dir"], "contract_sha256": images["contract_sha256"],
            "shape": list(images["targets"].shape),
            "candidate_counts": {key: len(value) for key, value in images["candidates"].items()},
            "gallery_norm_max_error": gallery_norm_error,
        },
        "code_contract": {
            "config_sha256": sha256(CONFIG),
            "model_sha256": sha256(ROOT / "src/d089_masked_retrieval.py"),
            "runner_sha256": sha256(Path(__file__)),
            "architecture_sha256": sha256(ROOT / "ARCHITECTURE.md"),
            "classification_head": False, "jepa": False, "ema_target_encoder": False,
        },
        "batch64_capacity": {
            "status": "PASS", "route": "separate_conv", "batch": logical_batch,
            "reconstruction_peak_allocated_gib": reconstruction_peak,
            "image_peak_allocated_gib": image_peak,
            "targets_and_gallery_resident_during_probe": True,
        },
        "smoke": smoke_result,
    }


def run(cfg: dict[str, Any], mode: str) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("D089 formal execution requires CUDA")
    device = torch.device("cuda")
    report = ROOT / cfg["report_dir"]
    run_root = ROOT / cfg["run_dir"]
    report.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True, exist_ok=True)
    data = load_subject_data(cfg)
    atomic_json(report / "split.json", split_evidence(data))
    images = load_image_targets(cfg, data, device)
    atomic_json(report / "target_contract.json", {
        "target_dir": images["target_dir"], "contract_sha256": images["contract_sha256"],
        "targets": list(images["targets"].shape), "gallery": list(images["gallery"].shape),
        "candidate_counts": {key: len(value) for key, value in images["candidates"].items()},
    })
    preflight = formal_preflight(cfg, data, images, device)
    atomic_json(report / "preflight.json", preflight)
    results = {}
    if mode in {"dev", "all"}:
        stats = fit_statistics(data["waveform"], data["fit"], device)
        for route in cfg["routes"]:
            status = {"status": "running", "phase": "development", "route": route, "pid": __import__("os").getpid()}
            atomic_json(report / "status.json", status)
            route_run = run_root / "development" / route
            completed_path = route_run / "result.json"
            if completed_path.exists():
                result = json.loads(completed_path.read_text(encoding="utf-8"))
                if result.get("contract") != checkpoint_contract(route, "development", data["fit"]):
                    raise AssertionError("Existing development result contract differs")
            else:
                pretrain_path = pretrain(route, "development", stats["waveform"], data["fit"],
                                         stats["frequency_mean"], stats["frequency_scale"], cfg, route_run)
                result = finetune(route, "development", pretrain_path, stats["waveform"], data["fit"],
                                  data["validation"], data, images, stats["frequency_mean"],
                                  stats["frequency_scale"], cfg, route_run, cfg["finetune"]["max_epochs"])
            atomic_json(route_run / "result.json", result)
            results[route] = result
        selection = {route: int(result["selected_epoch"]) for route, result in results.items()}
        atomic_json(report / "development_selection.json", selection)
        atomic_json(report / "development_results.json", results)
    if mode in {"final", "all"}:
        selection_path = report / "development_selection.json"
        if not selection_path.exists():
            raise FileNotFoundError("Development selection must exist before final refit")
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        stats = fit_statistics(data["waveform"], data["official_train"], device)
        final_results = {}
        for route in cfg["routes"]:
            atomic_json(report / "status.json", {"status": "running", "phase": "final_refit", "route": route,
                                                   "pid": __import__("os").getpid()})
            route_run = run_root / "final" / route
            completed_path = route_run / "result.json"
            if completed_path.exists():
                result = json.loads(completed_path.read_text(encoding="utf-8"))
                if result.get("contract") != checkpoint_contract(route, "final", data["official_train"]):
                    raise AssertionError("Existing final result contract differs")
            else:
                pretrain_path = pretrain(route, "final", stats["waveform"], data["official_train"],
                                         stats["frequency_mean"], stats["frequency_scale"], cfg, route_run)
                result = finetune(route, "final", pretrain_path, stats["waveform"], data["official_train"], None,
                                  data, images, stats["frequency_mean"], stats["frequency_scale"], cfg,
                                  route_run, int(selection[route]))
            atomic_json(route_run / "result.json", result)
            final_results[route] = result
        atomic_json(report / "final_results.json", final_results)
        results = final_results
    atomic_json(report / "status.json", {"status": "complete", "mode": mode})
    return {"status": "complete", "mode": mode, "results": results}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--run-dev", action="store_true")
    parser.add_argument("--run-final", action="store_true")
    parser.add_argument("--run-all", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    selected = sum((args.smoke, args.preflight, args.run_dev, args.run_final, args.run_all))
    if selected != 1:
        parser.error("select exactly one mode")
    if args.smoke:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        result = smoke(device)
        atomic_json(ROOT / cfg["report_dir"] / "smoke.json", result)
    elif args.preflight:
        if not torch.cuda.is_available():
            raise RuntimeError("D089 formal preflight requires CUDA")
        device = torch.device("cuda")
        data = load_subject_data(cfg)
        images = load_image_targets(cfg, data, device)
        result = formal_preflight(cfg, data, images, device)
        atomic_json(ROOT / cfg["report_dir"] / "preflight.json", result)
    else:
        mode = "all" if args.run_all else "dev" if args.run_dev else "final"
        result = run(cfg, mode)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
