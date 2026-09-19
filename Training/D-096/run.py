"""Train D096 on subject 0 first-30 rows and evaluate the last-20 endpoint once."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F


EXPERIMENT_DIR = Path(__file__).resolve().parent
ROOT = EXPERIMENT_DIR.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from eegdecoding.subject_data import load_subject_data, split_evidence  # noqa: E402
from model import (  # noqa: E402
    BRANCH_FEATURES,
    CHANNELS,
    CONCAT_FEATURES,
    DEPTH,
    DIM,
    D096CrossAttentionClassifier,
    FFT_SIZE,
    FFN_DIM,
    FREQUENCY_BINS,
    FREQUENCY_WINDOW_SAMPLES,
    FRONTEND_DIM,
    FUSED_FEATURES,
    HEADS,
    MULTISCALE_KERNELS,
    PATCHES,
    PATCH_SAMPLES,
    SPATIAL_BLOCKS,
    frequency_bin_centres,
    normalize_unit_sphere_coordinates,
    parameter_counts,
)


CONFIG = EXPERIMENT_DIR / "config/config.json"
GIB = 1024**3
EXPECTED_SUBJECT = 0
EXPECTED_CROP = [40, 440]
EXPECTED_CLASSES = 80
EXPECTED_TRAIN_SAMPLES = 2400
EXPECTED_TEST_SAMPLES = 1600
TEST_FORWARD_SAMPLES_BEFORE_EVALUATION = 0

Config = dict[str, Any]
PreparedData = dict[str, Any]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def workspace_relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    os.replace(temporary, path)


def append_jsonl(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_coordinates(cfg: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
    """Load, validate, normalize, and fingerprint the electrode coordinates."""
    path = EXPERIMENT_DIR / cfg["channel_map"]
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = sorted(csv.DictReader(stream), key=lambda row: int(row["tensor_index"]))
    if [int(row["tensor_index"]) for row in rows] != list(range(CHANNELS)):
        raise AssertionError("channel_map tensor order is not exactly 0..61")
    names = [str(row["canonical_name"]) for row in rows]
    if len(set(names)) != CHANNELS:
        raise AssertionError("channel_map canonical names are not unique")
    columns = cfg["coordinate_columns"]
    coordinates = torch.tensor(
        [[float(row[column]) for column in columns] for row in rows], dtype=torch.float32
    )
    normalized = normalize_unit_sphere_coordinates(coordinates)
    return normalized, {
        "path": workspace_relative(path),
        "file_sha256": sha256(path),
        "columns": columns,
        "axes": cfg["coordinate_axes"],
        "source": "MNE colin27_1005 template-fit approximation in channel_map; not subject digitization",
        "channel_names": names,
        "raw_coordinate_sha256": tensor_sha256(coordinates),
        "normalized_coordinate_sha256": tensor_sha256(normalized),
        "normalization": "row-wise r / ||r||_2; no centering and no axis change",
        "normalized_radius_min": float(normalized.norm(dim=-1).min()),
        "normalized_radius_max": float(normalized.norm(dim=-1).max()),
    }


def prepare_data(cfg: Config, model: D096CrossAttentionClassifier) -> PreparedData:
    """Load subject data and fit input normalization on the training split only."""
    data = load_subject_data(cfg)
    train_indices = data["official_train"]
    test_indices = data["test"]
    waveform = model.filter_input(data["waveform"])

    train_raw = waveform[train_indices]
    mean = train_raw.double().mean(dim=(0, 2)).float()
    variance = train_raw.double().var(dim=(0, 2), unbiased=False).float()
    scale = variance.clamp_min(1e-16).sqrt()
    model.set_input_statistics(mean, scale)

    train = ((train_raw - mean[None, :, None]) / scale[None, :, None]).contiguous()
    test_raw = waveform[test_indices]
    test = ((test_raw - mean[None, :, None]) / scale[None, :, None]).contiguous()
    try:
        train = train.pin_memory()
        test = test.pin_memory()
        pinned = True
    except RuntimeError:
        pinned = False
    return {
        "train": train,
        "train_labels": data["labels"][train_indices],
        "train_indices": train_indices,
        "test": test,
        "test_labels": data["labels"][test_indices],
        "test_indices": test_indices,
        "mean": mean,
        "scale": scale,
        "pinned": pinned,
        "data": data,
    }


def augment(value: torch.Tensor, cfg: dict[str, Any]) -> torch.Tensor:
    """Apply the configured waveform augmentations to one training batch."""
    settings = cfg["augmentation"]
    if (
        settings["gain_min"] == 1.0
        and settings["gain_max"] == 1.0
        and settings["noise_std_max"] == 0.0
        and settings["max_shift_samples"] == 0
        and settings["channel_dropout"] == 0.0
    ):
        return value
    output = value.clone()
    batch = len(output)
    gain = torch.empty((batch, 1, 1), device=output.device).uniform_(
        settings["gain_min"], settings["gain_max"]
    )
    output *= gain
    if settings["noise_std_max"] > 0:
        noise = torch.empty((batch, 1, 1), device=output.device).uniform_(
            0.0, settings["noise_std_max"]
        )
        output += torch.randn_like(output) * noise
    if settings["channel_dropout"] > 0:
        keep = torch.rand((batch, CHANNELS, 1), device=output.device) >= settings["channel_dropout"]
        output *= keep
    max_shift = int(settings["max_shift_samples"])
    if max_shift > 0:
        shifts = torch.randint(-max_shift, max_shift + 1, (batch,), device=output.device)
        for row, shift_value in enumerate(shifts.tolist()):
            if shift_value:
                output[row] = torch.roll(output[row], shift_value, dims=-1)
                if shift_value > 0:
                    output[row, :, :shift_value] = 0
                else:
                    output[row, :, shift_value:] = 0
    return output


def learning_rate(cfg: dict[str, Any], epoch: int) -> float:
    base = float(cfg["learning_rate"])
    minimum = float(cfg["minimum_learning_rate"])
    warmup = int(cfg["warmup_epochs"])
    epochs = int(cfg["epochs"])
    if epoch <= warmup:
        return base * epoch / max(1, warmup)
    progress = (epoch - warmup) / max(1, epochs - warmup)
    return minimum + 0.5 * (base - minimum) * (1.0 + math.cos(math.pi * progress))


def build_model(
    coordinates: torch.Tensor, cfg: Config, device: torch.device
) -> D096CrossAttentionClassifier:
    """Construct the D096 classifier from the experiment configuration."""
    return D096CrossAttentionClassifier(
        coordinates,
        classes=cfg["classes"],
        dropout=cfg["block_dropout"],
        head_dropout=cfg["head_dropout"],
        excluded_band_hz=cfg.get("excluded_band_hz"),
        post_fusion_projection=cfg.get("post_fusion_projection", False),
    ).to(device)


def build_optimizer(
    model: D096CrossAttentionClassifier, cfg: Config
) -> torch.optim.Optimizer:
    """Create the optimizer used by both probing and actual training."""
    return torch.optim.AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )


def train_epoch(
    model: D096CrossAttentionClassifier,
    waveform: torch.Tensor,
    labels: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    cfg: dict[str, Any],
    device: torch.device,
    generator: torch.Generator,
    microbatch: int,
) -> dict[str, float | int]:
    model.train()
    order = torch.randperm(len(waveform), generator=generator)
    loss_sum = 0.0
    correct = 0
    grad_norm = 0.0
    for logical_ids in order.split(int(cfg["logical_batch"])):
        optimizer.zero_grad(set_to_none=True)
        logical_size = len(logical_ids)
        for ids in logical_ids.split(microbatch):
            batch = augment(waveform[ids].to(device, non_blocking=True), cfg)
            target = labels[ids].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model.forward_standardized(batch)
                raw_loss = F.cross_entropy(
                    logits.float(), target, label_smoothing=float(cfg["label_smoothing"])
                )
                loss = raw_loss * (len(ids) / logical_size)
            loss.backward()
            loss_sum += float(raw_loss.detach()) * len(ids)
            correct += int((logits.argmax(1) == target).sum())
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"]))
        optimizer.step()
    return {
        "n": len(waveform),
        "loss": loss_sum / len(waveform),
        "acc_all": correct / len(waveform),
        "grad_norm": grad_norm,
    }


@torch.inference_mode()
def evaluate_test_once(
    model: D096CrossAttentionClassifier,
    waveform: torch.Tensor,
    labels: torch.Tensor,
    global_indices: torch.Tensor,
    data: dict[str, Any],
    cfg: dict[str, Any],
    device: torch.device,
    microbatch: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    loss_sum = 0.0
    correct = torch.zeros(cfg["classes"], dtype=torch.long)
    total = torch.zeros(cfg["classes"], dtype=torch.long)
    predictions: list[dict[str, Any]] = []
    for offset in range(0, len(waveform), microbatch):
        stop = min(offset + microbatch, len(waveform))
        batch = waveform[offset:stop].to(device, non_blocking=True)
        target = labels[offset:stop].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, auxiliary = model.forward_standardized(batch, return_aux=True)
        logits_float = logits.float()
        loss_sum += float(F.cross_entropy(logits_float, target, reduction="sum"))
        guesses = logits_float.argmax(1)
        probabilities = logits_float.softmax(1)
        confidence, top = probabilities.topk(5, dim=1)
        cross_attention = auxiliary["cross_attention_weights"]
        pooling = auxiliary["spatial_pooling_weights"]
        for local, truth, guess, choices, scores in zip(
            range(offset, stop), target.cpu(), guesses.cpu(), top.cpu(), confidence.cpu()
        ):
            label = int(truth)
            total[label] += 1
            correct[label] += int(truth == guess)
            global_index = int(global_indices[local])
            predictions.append({
                "test_local_index": local,
                "trial_id": str(data["trial_ids"][global_index]),
                "image_id": str(data["images"][global_index]),
                "source_index": int(data["source_indices"][global_index]),
                "label": label,
                "prediction": int(guess),
                "top5": [int(value) for value in choices],
                "top5_probability": [float(value) for value in scores],
                "cross_attention_entropy_mean": float(
                    (-(cross_attention[local - offset].float().clamp_min(1e-30).log()
                       * cross_attention[local - offset]).sum(-1)).mean()
                ),
                "cross_attention_diagonal_mass_mean": float(
                    cross_attention[local - offset].float().diagonal(dim1=-2, dim2=-1).mean()
                ),
                "spatial_pooling_entropy_mean": float(
                    (-(pooling[local - offset].float().clamp_min(1e-30).log() * pooling[local - offset]).sum(0)).mean()
                ),
            })
    per_class = correct.float() / total.clamp_min(1)
    metrics = {
        "n": int(total.sum()),
        "loss": loss_sum / int(total.sum()),
        "acc_all": float(correct.sum() / total.sum()),
        "correct": int(correct.sum()),
        "random_acc_all": 1.0 / cfg["classes"],
        "macro_class_accuracy": float(per_class.mean()),
        "per_class_accuracy": [float(value) for value in per_class],
        "test_forward_rounds": 1,
        "test_forward_samples": int(total.sum()),
    }
    return metrics, predictions


def probe_microbatch(
    coordinates: torch.Tensor,
    cfg: Config,
    device: torch.device,
) -> dict[str, Any]:
    """Find the largest candidate microbatch under the configured memory limit."""
    failures = []
    for candidate in cfg["microbatch_candidates"]:
        candidate = min(int(candidate), int(cfg["logical_batch"]))
        logits = None
        loss = None
        set_seed(int(cfg["seed"]) + 9500 + candidate)
        model = build_model(coordinates, cfg, device)
        optimizer = build_optimizer(model, cfg)
        waveform = torch.randn(candidate, CHANNELS, PATCHES * PATCH_SAMPLES, device=device)
        target = torch.arange(candidate, device=device) % cfg["classes"]
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model.forward_standardized(waveform)
                loss = F.cross_entropy(logits.float(), target, label_smoothing=cfg["label_smoothing"])
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize(device)
            peak = int(torch.cuda.max_memory_allocated(device))
            if peak <= float(cfg["peak_memory_limit_gib"]) * GIB:
                return {
                    "microbatch": candidate,
                    "peak_memory_bytes": peak,
                    "peak_memory_gib": peak / GIB,
                    "failed_larger_candidates": failures,
                }
            failures.append({"microbatch": candidate, "reason": "peak_limit", "peak_memory_bytes": peak})
        except torch.OutOfMemoryError:
            failures.append({"microbatch": candidate, "reason": "cuda_oom"})
        finally:
            del waveform, target, optimizer, model
            if logits is not None:
                del logits
            if loss is not None:
                del loss
            torch.cuda.empty_cache()
    raise RuntimeError(f"no D096 microbatch fits: {failures}")


def architecture_record(cfg: dict[str, Any]) -> dict[str, Any]:
    bins = frequency_bin_centres()
    return {
        "input": [CHANNELS, PATCHES * PATCH_SAMPLES],
        "sample_rate_hz": 1000,
        "crop_ms": [40, 440],
        "multiscale_convolution": {
            "per_electrode": True,
            "kernels_samples_and_ms": list(MULTISCALE_KERNELS),
            "features_per_branch": BRANCH_FEATURES,
            "concatenated_features": CONCAT_FEATURES,
            "fusion": f"grouped 1x1 Conv {CONCAT_FEATURES}->{FUSED_FEATURES}",
        },
        "waveform_tokens": {
            "patches": PATCHES,
            "patch_ms": PATCH_SAMPLES,
            "within_patch_reduction": "mean after temporal convolutions",
            "projection": f"Linear({FUSED_FEATURES},{FRONTEND_DIM})",
        },
        "frequency_branch": {
            "enabled": True,
            "alignment": "one 97 ms window centred on each 25 ms waveform token",
            "edge_padding": "reflect",
            "window": "periodic Hann",
            "fft_size": FFT_SIZE,
            "requested_range_hz": [12, 80],
            "retained_bin_centres_hz": [float(value) for value in bins],
            "retained_bins": FREQUENCY_BINS,
            "feature": "log(1 + |FFT|^2)",
            "projection": f"Linear({FREQUENCY_BINS},{FRONTEND_DIM})",
        },
        "cross_attention_fusion": {
            "direction": "waveform queries frequency",
            "scope": "16 temporal patches independently for each electrode",
            "heads": HEADS,
            "query_key_position": "electrode ID + unit-sphere coordinate + temporal position",
            "value": "frequency content only",
            "residual": "waveform + learned_scale * cross_attention_context",
            "residual_scale_initial": 0.1,
        },
        "embeddings": {
            "electrode_id": FRONTEND_DIM,
            "coordinate": f"unit sphere -> Linear(3,32) -> GELU -> Linear(32,{FRONTEND_DIM})",
            "temporal_position": FRONTEND_DIM,
            "trunk_projection": "none; fused tokens and embeddings already have trunk width",
        },
        "trunk": {
            "blocks": DEPTH,
            "dimension": DIM,
            "heads": HEADS,
            "order": ["depthwise temporal Conv", "global temporal attention", "optional spatial attention", "FFN"],
            "depthwise_temporal_kernel_tokens": 3,
            "spatial_attention_blocks": list(SPATIAL_BLOCKS),
            "ffn": [DIM, FFN_DIM, DIM],
        },
        "readout": {
            "spatial_pooling": "learned softmax weights, 62 electrodes -> 1 per time token",
            "pooled_shape": [PATCHES, DIM],
            "flattened": PATCHES * DIM,
            "head": f"LayerNorm -> Dropout -> Linear({PATCHES * DIM},{cfg['classes']})",
        },
    }


def save_checkpoint(
    path: Path,
    model: D096CrossAttentionClassifier,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    contract: dict[str, Any],
    prepared: dict[str, Any],
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "contract": contract,
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "input_mean": prepared["mean"],
        "input_scale": prepared["scale"],
    }, temporary)
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    """Parse the two intentionally exclusive runner modes."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG, help="JSON experiment configuration")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true", help="run one training update")
    mode.add_argument("--run", action="store_true", help="train and evaluate the endpoint")
    args = parser.parse_args()
    return args


def load_config(path: Path) -> Config:
    """Load the JSON configuration and validate the fixed D096 contract."""
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if (
        cfg["subject"] != EXPECTED_SUBJECT
        or cfg["crop"] != EXPECTED_CROP
        or cfg["classes"] != EXPECTED_CLASSES
    ):
        raise AssertionError("D096 subject/crop/class contract differs")
    if cfg["label_smoothing"] != 0.0:
        raise AssertionError("D096 requires ordinary cross entropy with label_smoothing=0")
    if cfg.get("frequency_branch") is not True:
        raise AssertionError("D096 requires the frequency branch for cross-attention")
    return cfg


def configure_cuda(cfg: Config) -> torch.device:
    """Validate the configured GPU lock and enable the D096 CUDA settings."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") != cfg["gpu_uuid"] or not torch.cuda.is_available():
        raise RuntimeError("D096 must use the configured local GPU UUID")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    return torch.device("cuda")


def prepare_output_dirs(cfg: Config) -> tuple[Path, Path]:
    """Create and return the run and report directories."""
    run_dir = EXPERIMENT_DIR / cfg["run_dir"]
    report_dir = EXPERIMENT_DIR / cfg["report_dir"]
    run_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, report_dir


def validate_split(prepared: PreparedData) -> dict[str, Any]:
    """Verify the fixed first-30/last-20 split before writing any artifacts."""
    split = split_evidence(prepared["data"])
    if split["official_train"] != EXPECTED_TRAIN_SAMPLES or split["test"] != EXPECTED_TEST_SAMPLES:
        raise AssertionError("D096 first30/last20 split count differs")
    if (
        split["per_class"]["official_train"] != [30] * EXPECTED_CLASSES
        or split["per_class"]["test"] != [20] * EXPECTED_CLASSES
    ):
        raise AssertionError("D096 per-class first30/last20 split differs")
    if split["official_train_test_image_overlap"] != 0:
        raise AssertionError("D096 train/test image IDs overlap")
    return split


def build_contract(
    cfg: Config,
    config_path: Path,
    coordinate_record: dict[str, Any],
    split: dict[str, Any],
) -> dict[str, Any]:
    """Build the provenance contract stored in checkpoints and reports."""
    return {
        "excluded_band_hz": cfg.get("excluded_band_hz"),
        "exclusion_protocol": cfg.get("exclusion_protocol"),
        "post_fusion_projection": cfg.get("post_fusion_projection", False),
        "task": cfg["task"],
        "config_path": workspace_relative(config_path),
        "config_sha256": sha256(config_path),
        "model_sha256": sha256(EXPERIMENT_DIR / "model.py"),
        "runner_sha256": sha256(Path(__file__)),
        "montage_sha256": coordinate_record["file_sha256"],
        "official_train_hash": split["official_train_hash"],
        "test_hash": split["test_hash"],
        "layout": "Training/D-096",
    }


def build_preflight(
    cfg: Config,
    model: D096CrossAttentionClassifier,
    prepared: PreparedData,
    coordinate_record: dict[str, Any],
    split: dict[str, Any],
    contract: dict[str, Any],
    microbatch_record: dict[str, Any],
) -> dict[str, Any]:
    """Collect the read-only checks and fingerprints emitted before training."""
    return {
        "status": "PASS",
        "contract": contract,
        "split": split,
        "architecture": architecture_record(cfg),
        "parameters": parameter_counts(model),
        "coordinates": coordinate_record,
        "statistics": {
            "fit_split": "official first 30 per class only",
            "mean_sha256": tensor_sha256(prepared["mean"]),
            "scale_sha256": tensor_sha256(prepared["scale"]),
        },
        "resident_data": {
            "train": list(prepared["train"].shape),
            "test": list(prepared["test"].shape),
            "pinned": prepared["pinned"],
        },
        "microbatch_probe": microbatch_record,
        "test_classifier_forward_samples": TEST_FORWARD_SAMPLES_BEFORE_EVALUATION,
    }


def run_smoke(
    model: D096CrossAttentionClassifier,
    prepared: PreparedData,
    cfg: Config,
    device: torch.device,
    microbatch: int,
    report_dir: Path,
) -> int:
    """Run one forward/backward/update cycle and write the smoke report."""
    model.train()
    optimizer = build_optimizer(model, cfg)
    batch = prepared["train"][:microbatch].to(device)
    target = prepared["train_labels"][:microbatch].to(device)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits, auxiliary = model.forward_standardized(batch, return_aux=True)
        loss = F.cross_entropy(logits.float(), target, label_smoothing=0.0)
    loss.backward()
    optimizer.step()

    cross_attention = auxiliary["cross_attention_weights"]
    smoke = {
        "status": "PASS",
        "loss": float(loss.detach()),
        "logits_shape": list(logits.shape),
        "finite": bool(torch.isfinite(logits).all()),
        "cross_attention_residual_scale_after_first_update": float(
            model.cross_modal_fusion.residual_scale.detach()
        ),
        "train_mode_cross_attention_row_sum_max_error_after_dropout": float(
            (cross_attention.detach().float().sum(-1) - 1.0).abs().max()
        ),
        "spatial_weight_sum_max_error": float(
            (auxiliary["spatial_pooling_weights"].detach().float().sum(1) - 1.0).abs().max()
        ),
        "microbatch": microbatch,
        "test_classifier_forward_samples": TEST_FORWARD_SAMPLES_BEFORE_EVALUATION,
    }
    atomic_json(report_dir / "smoke.json", smoke)
    print(json.dumps(smoke, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


def train_model(
    model: D096CrossAttentionClassifier,
    prepared: PreparedData,
    cfg: Config,
    device: torch.device,
    microbatch: int,
    run_dir: Path,
    report_dir: Path,
    contract: dict[str, Any],
) -> tuple[list[dict[str, Any]], float]:
    """Train for the fixed budget, checkpointing and logging every epoch."""
    optimizer = build_optimizer(model, cfg)
    generator = torch.Generator().manual_seed(cfg["seed"] + 9501)
    curve_path = run_dir / "train_curve.jsonl"
    if curve_path.exists():
        raise FileExistsError(f"refusing to append to existing D096 curve: {curve_path}")

    started = time.perf_counter()
    history: list[dict[str, Any]] = []
    atomic_json(
        report_dir / "status.json",
        {
            "status": "training",
            "epoch": 0,
            "test_classifier_forward_samples": TEST_FORWARD_SAMPLES_BEFORE_EVALUATION,
        },
    )
    for epoch in range(1, int(cfg["epochs"]) + 1):
        lr = learning_rate(cfg, epoch)
        for group in optimizer.param_groups:
            group["lr"] = lr
        metrics = train_epoch(
            model,
            prepared["train"],
            prepared["train_labels"],
            optimizer,
            cfg,
            device,
            generator,
            microbatch,
        )
        row = {
            "epoch": epoch,
            "train": metrics,
            "learning_rate": lr,
            "elapsed_seconds": time.perf_counter() - started,
            "test_classifier_forward_samples": TEST_FORWARD_SAMPLES_BEFORE_EVALUATION,
        }
        history.append(row)
        append_jsonl(curve_path, row)
        save_checkpoint(run_dir / "latest.pt", model, optimizer, epoch, contract, prepared)
        if epoch % int(cfg["checkpoint_every_epochs"]) == 0 or epoch == int(cfg["epochs"]):
            save_checkpoint(run_dir / f"epoch{epoch:03d}.pt", model, optimizer, epoch, contract, prepared)
        atomic_json(report_dir / "status.json", {"status": "training", **row})
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train_loss": metrics["loss"],
                    "train_acc_all": metrics["acc_all"],
                    "lr": lr,
                    "elapsed_seconds": row["elapsed_seconds"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return history, started


def write_predictions(path: Path, predictions: list[dict[str, Any]]) -> None:
    """Write endpoint predictions without allowing accidental overwrites."""
    with path.open("x", encoding="utf-8") as stream:
        for row in predictions:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def build_result(
    model: D096CrossAttentionClassifier,
    cfg: Config,
    history: list[dict[str, Any]],
    test_metrics: dict[str, Any],
    started: float,
    microbatch: int,
    run_dir: Path,
    prediction_path: Path,
    contract: dict[str, Any],
) -> dict[str, Any]:
    """Build the final result record after the single endpoint evaluation."""
    return {
        "status": "complete",
        "subject": EXPECTED_SUBJECT,
        "epochs": cfg["epochs"],
        "final_train": history[-1]["train"],
        "test": test_metrics,
        "elapsed_seconds": time.perf_counter() - started,
        "parameters": parameter_counts(model),
        "microbatch": microbatch,
        "checkpoint": workspace_relative(run_dir / f"epoch{int(cfg['epochs']):03d}.pt"),
        "predictions": workspace_relative(prediction_path),
        "contract": contract,
        "selection": cfg["selection"],
    }


def main() -> int:
    args = parse_args()

    config_path = args.config.resolve()
    cfg = load_config(config_path)
    device = configure_cuda(cfg)
    run_dir, report_dir = prepare_output_dirs(cfg)

    coordinates, coordinate_record = load_coordinates(cfg)
    microbatch_record = probe_microbatch(coordinates, cfg, device)
    set_seed(cfg["seed"])
    model = build_model(coordinates, cfg, device)
    prepared = prepare_data(cfg, model)
    split = validate_split(prepared)
    contract = build_contract(cfg, config_path, coordinate_record, split)
    preflight = build_preflight(
        cfg, model, prepared, coordinate_record, split, contract, microbatch_record
    )
    atomic_json(report_dir / "preflight.json", preflight)

    microbatch = int(microbatch_record["microbatch"])
    if args.smoke:
        return run_smoke(model, prepared, cfg, device, microbatch, report_dir)

    history, started = train_model(
        model, prepared, cfg, device, microbatch, run_dir, report_dir, contract
    )

    test_metrics, predictions = evaluate_test_once(
        model,
        prepared["test"],
        prepared["test_labels"],
        prepared["test_indices"],
        prepared["data"],
        cfg,
        device,
        microbatch,
    )
    prediction_path = run_dir / "test_predictions.jsonl"
    write_predictions(prediction_path, predictions)
    result = build_result(
        model,
        cfg,
        history,
        test_metrics,
        started,
        microbatch,
        run_dir,
        prediction_path,
        contract,
    )
    atomic_json(run_dir / "result.json", result)
    atomic_json(report_dir / "status.json", result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
