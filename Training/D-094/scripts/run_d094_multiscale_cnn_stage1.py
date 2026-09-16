"""Train the D094 multi-scale temporal/spatial CNN on subject 0 development data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from eegdecoding.subject_data import load_subject_data, split_evidence  # noqa: E402
from d094_multiscale_cnn import (  # noqa: E402
    CHANNELS,
    D094Stage1Classifier,
    SEGMENTS,
    SPATIAL_FEATURES,
    TARGET_SAMPLES,
    TEMPORAL_KERNEL_MS,
    TEMPORAL_KERNEL_SAMPLES,
    parameter_counts,
)


CONFIG = ROOT / "config/d094_multiscale_cnn_stage1.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def append_jsonl(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


@torch.inference_mode()
def resample_split(
    model: D094Stage1Classifier,
    raw: torch.Tensor,
    indices: torch.Tensor,
    device: torch.device,
    batch_size: int = 128,
) -> torch.Tensor:
    output = torch.empty((len(indices), CHANNELS, TARGET_SAMPLES), dtype=torch.float32)
    model.eval()
    for start in range(0, len(indices), batch_size):
        ids = indices[start:start + batch_size]
        value = raw[ids].to(device, non_blocking=True)
        output[start:start + len(ids)] = model.resampler(value).float().cpu()
    return output


def prepare_development_data(cfg: dict, model: D094Stage1Classifier, device: torch.device):
    data = load_subject_data(cfg)
    # Resample fit first, fit normalization only there, and then transform the
    # validation split.  Official test waveforms are never passed to the model.
    fit_raw = resample_split(model, data["waveform"], data["fit"], device)
    mean = fit_raw.double().mean(dim=(0, 2)).float()
    variance = fit_raw.double().var(dim=(0, 2), unbiased=False).float()
    scale = variance.clamp_min(1e-16).sqrt()
    model.set_input_statistics(mean, scale)
    fit = ((fit_raw - mean[None, :, None]) / scale[None, :, None]).contiguous()
    validation_raw = resample_split(model, data["waveform"], data["validation"], device)
    validation = ((validation_raw - mean[None, :, None]) / scale[None, :, None]).contiguous()
    try:
        fit = fit.pin_memory()
        validation = validation.pin_memory()
        pinned = True
    except RuntimeError:
        pinned = False
    return {
        "fit": fit,
        "fit_labels": data["labels"][data["fit"]],
        "validation": validation,
        "validation_labels": data["labels"][data["validation"]],
        "validation_global_indices": data["validation"],
        "data": data,
        "mean": mean,
        "scale": scale,
        "pinned": pinned,
    }


def augment(value: torch.Tensor, cfg: dict) -> torch.Tensor:
    settings = cfg["augmentation"]
    output = value.clone()
    batch = len(output)
    gain = torch.empty((batch, 1, 1), device=output.device).uniform_(settings["gain_min"], settings["gain_max"])
    output *= gain
    noise = torch.empty((batch, 1, 1), device=output.device).uniform_(0.0, settings["noise_std_max"])
    output += torch.randn_like(output) * noise
    keep = torch.rand((batch, CHANNELS, 1), device=output.device) >= settings["channel_dropout"]
    output *= keep
    max_shift = int(settings["max_shift_250hz_samples"])
    shifts = torch.randint(-max_shift, max_shift + 1, (batch,), device=output.device)
    for row, shift_value in enumerate(shifts.tolist()):
        if shift_value:
            output[row] = torch.roll(output[row], shift_value, dims=-1)
            if shift_value > 0:
                output[row, :, :shift_value] = 0
            else:
                output[row, :, shift_value:] = 0
    return output


def train_epoch(model, waveform, labels, optimizer, cfg, device, generator):
    model.train()
    order = torch.randperm(len(waveform), generator=generator)
    loss_sum = 0.0
    correct = 0
    last_grad_norm = 0.0
    for ids in order.split(cfg["logical_batch"]):
        batch = augment(waveform[ids].to(device, non_blocking=True), cfg)
        target = labels[ids].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model.forward_resampled(batch)
            loss = F.cross_entropy(logits.float(), target, label_smoothing=cfg["label_smoothing"])
        loss.backward()
        last_grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"]))
        optimizer.step()
        loss_sum += float(loss.detach()) * len(ids)
        correct += int((logits.argmax(1) == target).sum())
    return {
        "n": len(waveform),
        "loss": loss_sum / len(waveform),
        "acc_all": correct / len(waveform),
        "grad_norm": last_grad_norm,
    }


@torch.inference_mode()
def evaluate(model, waveform, labels, cfg, device, predictions=False):
    model.eval()
    loss_sum = 0.0
    correct = torch.zeros(cfg["classes"], dtype=torch.long)
    total = torch.zeros(cfg["classes"], dtype=torch.long)
    rows = []
    for offset in range(0, len(waveform), cfg["logical_batch"]):
        batch = waveform[offset:offset + cfg["logical_batch"]].to(device, non_blocking=True)
        target = labels[offset:offset + cfg["logical_batch"]].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model.forward_resampled(batch)
        loss_sum += float(F.cross_entropy(logits.float(), target, reduction="sum"))
        guesses = logits.argmax(1)
        for truth, guess in zip(target.cpu(), guesses.cpu()):
            total[int(truth)] += 1
            correct[int(truth)] += int(truth == guess)
        if predictions:
            probabilities = logits.float().softmax(1)
            confidence, top = probabilities.topk(5, dim=1)
            for local, truth, choices, scores in zip(
                range(offset, offset + len(target)), target.cpu(), top.cpu(), confidence.cpu()
            ):
                rows.append({
                    "validation_local_index": local,
                    "label": int(truth),
                    "top5": [int(value) for value in choices],
                    "top5_probability": [float(value) for value in scores],
                })
    per_class = correct.float() / total.clamp_min(1)
    metrics = {
        "n": int(total.sum()),
        "loss": loss_sum / int(total.sum()),
        "acc_all": float(correct.sum() / total.sum()),
        "macro_class_accuracy": float(per_class.mean()),
        "per_class_accuracy": [float(value) for value in per_class],
        "random_acc_all": 1.0 / cfg["classes"],
    }
    return metrics, rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not args.run and not args.smoke:
        parser.error("choose --smoke or --run")
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    if os.environ.get("CUDA_VISIBLE_DEVICES") != cfg["gpu_uuid"] or not torch.cuda.is_available():
        raise RuntimeError("D094 must use the configured local GPU UUID")
    device = torch.device("cuda")
    run_dir = ROOT / cfg["run_dir"]
    report_dir = ROOT / cfg["report_dir"]
    run_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])
    model = D094Stage1Classifier(cfg["classes"], cfg["hidden"], cfg["dropout"]).to(device)
    prepared = prepare_development_data(cfg, model, device)
    split = split_evidence(prepared["data"])
    contract = {
        "task": cfg["task"],
        "config_sha256": sha256(CONFIG),
        "model_sha256": sha256(ROOT / "src/d094_multiscale_cnn.py"),
        "runner_sha256": sha256(Path(__file__)),
        "fit_hash": split["fit_hash"],
        "validation_hash": split["validation_hash"],
        "official_train_hash": split["official_train_hash"],
        "test_hash": split["test_hash"],
    }
    preflight = {
        "status": "PASS",
        "contract": contract,
        "split": split,
        "architecture": {
            "input": [CHANNELS, 400],
            "resample": "fixed 33-tap windowed-sinc anti-alias filter; 1000 Hz -> 250 Hz",
            "resampled_shape": [CHANNELS, TARGET_SAMPLES],
            "multiscale_ms": list(TEMPORAL_KERNEL_MS),
            "multiscale_samples": list(TEMPORAL_KERNEL_SAMPLES),
            "temporal_grouping": "four Conv1d branches, groups=62; no pre-spatial electrode mixing",
            "spatial": "Conv2d kernel=(62,1), four scale inputs -> 64 features",
            "depthwise_temporal_blocks": 2,
            "post_resample_downsampling": False,
            "segments": SEGMENTS,
            "segment_statistics": ["mean", "variance"],
            "mlp_hidden": cfg["hidden"],
            "classes": cfg["classes"],
        },
        "parameters": parameter_counts(model),
        "statistics": {
            "fit_only": True,
            "mean_sha256": tensor_sha256(prepared["mean"]),
            "scale_sha256": tensor_sha256(prepared["scale"]),
        },
        "resident_data": {
            "fit": list(prepared["fit"].shape),
            "validation": list(prepared["validation"].shape),
            "pinned": prepared["pinned"],
        },
        "test_classifier_forward_count": 0,
    }
    atomic_json(report_dir / "preflight.json", preflight)
    if args.smoke:
        batch = prepared["fit"][:2].to(device)
        target = prepared["fit_labels"][:2].to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"])
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model.forward_resampled(batch)
            loss = F.cross_entropy(logits.float(), target)
        loss.backward()
        optimizer.step()
        smoke = {
            "status": "PASS",
            "loss": float(loss.detach()),
            "logits_shape": list(logits.shape),
            "finite": bool(torch.isfinite(logits).all()),
            "test_classifier_forward_count": 0,
        }
        atomic_json(report_dir / "smoke.json", smoke)
        print(json.dumps(smoke, sort_keys=True), flush=True)
        return 0

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=cfg["lr_patience"],
        min_lr=cfg["minimum_learning_rate"],
    )
    generator = torch.Generator().manual_seed(cfg["seed"] + 9401)
    curve_path = run_dir / "development_curve.jsonl"
    best_path = run_dir / "development_best.pt"
    best_key = None
    best_epoch = 0
    stale = 0
    history = []
    started = time.perf_counter()
    atomic_json(report_dir / "status.json", {
        "status": "development_training", "epoch": 0, "test_classifier_forward_count": 0
    })
    for epoch in range(1, cfg["max_epochs"] + 1):
        train = train_epoch(
            model, prepared["fit"], prepared["fit_labels"], optimizer, cfg, device, generator
        )
        validation, _ = evaluate(
            model, prepared["validation"], prepared["validation_labels"], cfg, device
        )
        key = (validation["acc_all"], -validation["loss"])
        improved = best_key is None or key > best_key
        stale = 0 if improved else stale + 1
        if improved:
            best_key = key
            best_epoch = epoch
            torch.save({
                "contract": contract,
                "epoch": epoch,
                "model": model.state_dict(),
                "validation": validation,
                "input_mean": prepared["mean"],
                "input_scale": prepared["scale"],
            }, best_path)
        scheduler.step(validation["loss"])
        row = {
            "epoch": epoch,
            "train": train,
            "validation": validation,
            "improved": improved,
            "stale": stale,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "elapsed_seconds": time.perf_counter() - started,
            "test_classifier_forward_count": 0,
        }
        history.append(row)
        append_jsonl(curve_path, row)
        torch.save({
            "contract": contract,
            "epoch": epoch,
            "best_epoch": best_epoch,
            "best_key": list(best_key),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "history": history,
        }, run_dir / "development_latest.tmp")
        os.replace(run_dir / "development_latest.tmp", run_dir / "development_latest.pt")
        status = {
            "status": "development_training",
            "epoch": epoch,
            "train": train,
            "validation": validation,
            "best_epoch": best_epoch,
            "best_acc_all": best_key[0],
            "stale": stale,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "test_classifier_forward_count": 0,
        }
        atomic_json(report_dir / "status.json", status)
        print(json.dumps({
            "epoch": epoch,
            "train_acc": train["acc_all"],
            "val_acc": validation["acc_all"],
            "val_loss": validation["loss"],
            "best_epoch": best_epoch,
            "lr": optimizer.param_groups[0]["lr"],
        }, sort_keys=True), flush=True)
        if stale >= cfg["early_stop_patience"]:
            break

    saved = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(saved["model"], strict=True)
    validation, predictions = evaluate(
        model,
        prepared["validation"],
        prepared["validation_labels"],
        cfg,
        device,
        predictions=True,
    )
    data = prepared["data"]
    with (run_dir / "development_validation_predictions.jsonl").open("w", encoding="utf-8") as stream:
        for row, global_index in zip(predictions, prepared["validation_global_indices"].tolist()):
            row.update({
                "trial_id": str(data["trial_ids"][global_index]),
                "image_id": str(data["images"][global_index]),
                "source_index": int(data["source_indices"][global_index]),
            })
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    result = {
        "status": "development_complete",
        "subject": cfg["subject"],
        "selected_epoch": best_epoch,
        "development_validation": validation,
        "training_epochs_run": len(history),
        "elapsed_seconds": time.perf_counter() - started,
        "parameters": parameter_counts(model),
        "checkpoint": str(best_path),
        "predictions": str(run_dir / "development_validation_predictions.jsonl"),
        "contract": contract,
        "test_classifier_forward_count": 0,
        "test_status": "sealed_not_evaluated",
    }
    atomic_json(run_dir / "result.json", result)
    atomic_json(report_dir / "status.json", result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
