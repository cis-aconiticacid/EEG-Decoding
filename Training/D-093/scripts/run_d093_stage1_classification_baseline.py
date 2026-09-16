"""Audit, develop, refit, and once-only test the D093 Stage-1 classifier."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_d089_corrected_masked_retrieval as common  # noqa: E402
from d089_masked_retrieval import CHANNELS, D089Encoder, periodic_hann_log_psd  # noqa: E402
from d093_stage1_classifier import (  # noqa: E402
    Stage1Classifier,
    enable_sdpa_temporal_attention,
    parameter_counts,
)


CONFIG = ROOT / "config/d093_stage1_classification_baseline.json"
STOP = False


def request_stop(*_args) -> None:
    global STOP
    STOP = True


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


def waveform_statistics(waveform: torch.Tensor, fit: torch.Tensor):
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
    standardized = ((waveform - mean.float()[None, :, None]) / scale.float()[None, :, None]).contiguous()
    return mean.float(), scale.float(), standardized


def frequency_statistics(standardized: torch.Tensor, fit: torch.Tensor):
    total = torch.zeros((CHANNELS, 32), dtype=torch.float64)
    square = torch.zeros((CHANNELS, 32), dtype=torch.float64)
    count = 0
    for indices in fit.split(32):
        value = periodic_hann_log_psd(standardized[indices]).double()
        total += value.sum(0)
        square += value.square().sum(0)
        count += len(indices)
    mean = total / count
    scale = (square / count - mean.square()).clamp_min(1e-12).sqrt()
    return mean.float(), scale.float()


def load_pretrained_encoder(cfg: dict) -> tuple[D089Encoder, dict]:
    path = ROOT / cfg["pretrain_best"]
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved.get("route") != cfg["route"] or int(saved.get("step", -1)) != cfg["pretrain_expected_step"]:
        raise RuntimeError("Pretraining checkpoint identity mismatch")
    state = {
        name.removeprefix("encoder."): value
        for name, value in saved["model"].items()
        if name.startswith("encoder.")
    }
    encoder = D089Encoder(
        cfg["route"],
        dropout=0.1,
        activation_checkpointing=True,
        frequency_fusion_scale=cfg["frequency_fusion_scale"],
    )
    encoder.load_state_dict(state, strict=True)
    patched = enable_sdpa_temporal_attention(encoder)
    if patched != len(encoder.blocks):
        raise RuntimeError("Not every temporal attention block uses SDPA")
    return encoder, saved


def build_model(cfg: dict, device: torch.device) -> tuple[Stage1Classifier, dict]:
    torch.manual_seed(cfg["seed"] + 9300)
    torch.cuda.manual_seed_all(cfg["seed"] + 9300)
    encoder, pretrain = load_pretrained_encoder(cfg)
    model = Stage1Classifier(encoder, cfg["classes"], cfg["head_dropout"]).to(device)
    return model, pretrain


def set_phase(model: Stage1Classifier, phase: str, unfreeze_blocks: int) -> None:
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(False)
    if phase == "encoder_tail":
        for block in model.encoder.blocks[-unfreeze_blocks:]:
            for parameter in block.parameters():
                parameter.requires_grad_(True)
        for module in (model.encoder.final_norm, model.encoder.frequency_q_norm, model.encoder.frequency_kv_norm):
            for parameter in module.parameters():
                parameter.requires_grad_(True)
    for module in (model.norm, model.head):
        for parameter in module.parameters():
            parameter.requires_grad_(True)


def optimizer_groups(model: Stage1Classifier, cfg: dict, reductions: int):
    groups = []
    factor = 0.5 ** reductions
    for encoder_group, base_lr, role in (
        (True, cfg["encoder_lr"], "encoder"),
        (False, cfg["head_lr"], "head"),
    ):
        decay, no_decay = [], []
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad or name.startswith("encoder.") != encoder_group:
                continue
            target = no_decay if parameter.ndim == 1 or name.endswith(".bias") or "norm" in name.lower() else decay
            target.append(parameter)
        if decay:
            groups.append({"params": decay, "lr": base_lr * factor, "weight_decay": cfg["weight_decay"], "role": role})
        if no_decay:
            groups.append({"params": no_decay, "lr": base_lr * factor, "weight_decay": 0.0, "role": role})
    return groups


def augment(waveform: torch.Tensor, cfg: dict) -> torch.Tensor:
    settings = cfg["augmentation"]
    value = waveform.clone()
    batch = len(value)
    gain = torch.empty((batch, 1, 1), device=value.device).uniform_(settings["gain_min"], settings["gain_max"])
    value *= gain
    noise = torch.empty((batch, 1, 1), device=value.device).uniform_(0.0, settings["noise_std_max"])
    value += torch.randn_like(value) * noise
    keep = torch.rand((batch, CHANNELS, 1), device=value.device) >= settings["channel_dropout"]
    value *= keep
    max_shift = int(settings["max_shift_samples"])
    shifts = torch.randint(-max_shift, max_shift + 1, (batch,), device=value.device)
    for row, shift_value in enumerate(shifts):
        shift = int(shift_value)
        if shift:
            value[row] = torch.roll(value[row], shift, dims=-1)
            if shift > 0:
                value[row, :, :shift] = 0
            else:
                value[row, :, shift:] = 0
    return value


@torch.inference_mode()
def evaluate(model, waveform, labels, indices, frequency_mean, frequency_scale, microbatch, device, predictions=False):
    model.eval()
    loss_sum = 0.0
    correct = torch.zeros(80, dtype=torch.long)
    total = torch.zeros(80, dtype=torch.long)
    output_rows = []
    for ids in indices.split(microbatch):
        batch = waveform[ids].to(device, non_blocking=True)
        target = labels[ids].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(batch, frequency_mean, frequency_scale)
        loss_sum += float(F.cross_entropy(logits.float(), target, reduction="sum"))
        predicted = logits.argmax(1)
        for truth, guess in zip(target.cpu(), predicted.cpu()):
            total[int(truth)] += 1
            correct[int(truth)] += int(truth == guess)
        if predictions:
            probabilities = logits.float().softmax(1)
            confidence, top = probabilities.topk(5, dim=1)
            for index, truth, guesses, scores in zip(ids, target.cpu(), top.cpu(), confidence.cpu()):
                output_rows.append({
                    "local_index": int(index), "label": int(truth),
                    "top5": [int(value) for value in guesses],
                    "top5_probability": [float(value) for value in scores],
                })
    n = int(total.sum())
    per_class = correct.float() / total.clamp_min(1)
    result = {
        "n": n,
        "loss": loss_sum / n,
        "acc_all": float(correct.sum() / n),
        "macro_class_accuracy": float(per_class.mean()),
        "per_class_accuracy": [float(value) for value in per_class],
        "random_acc_all": 1.0 / 80.0,
    }
    return result, output_rows


def train_epoch(model, waveform, labels, indices, frequency_mean, frequency_scale, optimizer, cfg, microbatch, device, rng):
    model.train()
    if not any(parameter.requires_grad for parameter in model.encoder.parameters()):
        model.encoder.eval()
    order = indices[torch.from_numpy(rng.permutation(len(indices))).long()]
    loss_sum = 0.0
    correct = 0
    seen = 0
    last_grad_norm = 0.0
    for logical in order.split(cfg["logical_batch"]):
        optimizer.zero_grad(set_to_none=True)
        for ids in logical.split(microbatch):
            batch = augment(waveform[ids].to(device, non_blocking=True), cfg)
            target = labels[ids].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(batch, frequency_mean, frequency_scale)
                loss = F.cross_entropy(logits.float(), target, label_smoothing=cfg["label_smoothing"])
            (loss * (len(ids) / len(logical))).backward()
            loss_sum += float(loss.detach()) * len(ids)
            correct += int((logits.argmax(1) == target).sum())
            seen += len(ids)
        last_grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"]))
        optimizer.step()
    return {"loss": loss_sum / seen, "acc_all": correct / seen, "grad_norm": last_grad_norm, "n": seen}


def capacity_probe(cfg, waveform, labels, fit_indices, frequency_mean, frequency_scale, device):
    attempts = []
    for candidate in cfg["microbatch_candidates"]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        try:
            model, _ = build_model(cfg, device)
            set_phase(model, "encoder_tail", cfg["phase_b_unfreeze_blocks"])
            optimizer = torch.optim.AdamW(optimizer_groups(model, cfg, 0))
            ids = fit_indices[:candidate]
            batch = waveform[ids].to(device)
            target = labels[ids].to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = F.cross_entropy(model(batch, frequency_mean, frequency_scale).float(), target)
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize(device)
            peak = torch.cuda.max_memory_allocated(device) / 2**30
            attempts.append({"microbatch": candidate, "status": "PASS", "peak_allocated_gib": peak})
            del model, optimizer, batch, target, loss
            torch.cuda.empty_cache()
            if peak <= cfg["peak_memory_limit_gib"]:
                return candidate, attempts
        except torch.cuda.OutOfMemoryError:
            attempts.append({"microbatch": candidate, "status": "OOM"})
            torch.cuda.empty_cache()
    raise RuntimeError(f"No microbatch passed: {attempts}")


def monitor_gpu(cfg: dict, path: Path, stopped: threading.Event) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("timestamp,gpu_util,memory_used_mib,memory_total_mib,temperature_c,power_w\n", encoding="utf-8")
    while not stopped.is_set():
        result = subprocess.run([
            "nvidia-smi", f"--id={cfg['gpu_uuid']}",
            "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ], capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            with path.open("a", encoding="utf-8") as stream:
                stream.write(f"{time.time()},{result.stdout.strip()}\n")
        stopped.wait(30)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        parser.error("--run is required")
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    if os.environ.get("CUDA_VISIBLE_DEVICES") != cfg["gpu_uuid"] or not torch.cuda.is_available():
        raise RuntimeError("D093 must use the configured local GPU UUID")
    device = torch.device("cuda")
    report_dir = ROOT / cfg["report_dir"]
    run_dir = ROOT / cfg["run_dir"]
    report_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])
    data = common.load_subject_data(cfg)
    wave_mean, wave_scale, waveform = waveform_statistics(data["waveform"], data["fit"])
    frequency_mean, frequency_scale = frequency_statistics(waveform, data["fit"])
    try:
        waveform = waveform.pin_memory()
        pinned = True
    except RuntimeError:
        pinned = False
    frequency_mean = frequency_mean.to(device)
    frequency_scale = frequency_scale.to(device)
    labels = data["labels"]
    microbatch, attempts = capacity_probe(
        cfg, waveform, labels, data["fit"], frequency_mean, frequency_scale, device
    )

    _, pretrain = load_pretrained_encoder(cfg)
    split = common.split_evidence(data)
    contract = {
        "task": cfg["task"],
        "config_sha256": sha256(CONFIG),
        "model_sha256": sha256(ROOT / "src/d093_stage1_classifier.py"),
        "runner_sha256": sha256(Path(__file__)),
        "pretrain_sha256": sha256(ROOT / cfg["pretrain_best"]),
        "pretrain_step": int(pretrain["step"]),
        "fit_hash": split["fit_hash"],
        "validation_hash": split["validation_hash"],
        "official_train_hash": split["official_train_hash"],
        "test_hash": split["test_hash"],
    }
    model, _ = build_model(cfg, device)
    set_phase(model, "head_only", cfg["phase_b_unfreeze_blocks"])
    preflight = {
        "status": "PASS",
        "contract": contract,
        "split": split,
        "input": {"shape": list(waveform.shape), "resident_in_system_ram": True, "pinned": pinned},
        "statistics": {
            "wave_mean_sha256": tensor_sha256(wave_mean), "wave_scale_sha256": tensor_sha256(wave_scale),
            "frequency_mean_sha256": tensor_sha256(frequency_mean), "frequency_scale_sha256": tensor_sha256(frequency_scale),
        },
        "architecture": {
            "encoder": "D089 local_conv, 8 blocks, 384 width, 8 heads",
            "frequency_fusion_scale": cfg["frequency_fusion_scale"],
            "pool": "mean over 62 channels; preserve and flatten 8 time patches",
            "head": "LayerNorm(3072), Dropout(0.2), Linear(3072,80)",
            "attention_backend": cfg["attention_backend"],
            "forbidden_components": ["image_embeddings", "image_queries", "latent_regression", "retrieval", "JEPA"],
        },
        "capacity_attempts": attempts,
        "parameters": parameter_counts(model),
        "microbatch": microbatch,
        "logical_batch": cfg["logical_batch"],
        "test_forward_count": 0,
    }
    atomic_json(report_dir / "preflight.json", preflight)

    phase = "head_only"
    reductions = 0
    stale = 0
    set_phase(model, phase, cfg["phase_b_unfreeze_blocks"])
    optimizer = torch.optim.AdamW(optimizer_groups(model, cfg, reductions))
    rng = np.random.default_rng(cfg["seed"] + 9301)
    history = []
    best_key = None
    best_epoch = 0
    best_path = run_dir / "development_best.pt"
    start = time.perf_counter()
    stopped = threading.Event()
    monitor = threading.Thread(target=monitor_gpu, args=(cfg, report_dir / "gpu_samples.csv", stopped), daemon=True)
    monitor.start()
    atomic_json(report_dir / "status.json", {
        "status": "development_training", "epoch": 0, "phase": phase, "test_forward_count": 0,
        "microbatch": microbatch,
    })
    try:
        for epoch in range(1, cfg["max_epochs_safety_cap"] + 1):
            if STOP:
                break
            train = train_epoch(
                model, waveform, labels, data["fit"], frequency_mean, frequency_scale,
                optimizer, cfg, microbatch, device, rng,
            )
            validation, _ = evaluate(
                model, waveform, labels, data["validation"], frequency_mean, frequency_scale,
                microbatch, device,
            )
            key = (validation["acc_all"], -validation["loss"])
            improved = best_key is None or key > best_key
            stale = 0 if improved else stale + 1
            row = {
                "stage": "development", "epoch": epoch, "phase": phase, "train": train,
                "validation": validation, "selection_key": list(key), "improved": improved,
                "stale": stale, "lr_reductions": reductions,
                "learning_rates": {group["role"]: group["lr"] for group in optimizer.param_groups},
                "elapsed_seconds": time.perf_counter() - start, "test_forward_count": 0,
            }
            if improved:
                best_key = key
                best_epoch = epoch
                torch.save({
                    "contract": contract, "epoch": epoch, "phase": phase, "model": model.state_dict(),
                    "validation": validation, "selection_key": list(key),
                }, best_path)

            if phase == "head_only" and epoch >= cfg["phase_a_min_epochs"] and stale >= cfg["phase_a_patience"]:
                phase = "encoder_tail"
                stale = 0
                set_phase(model, phase, cfg["phase_b_unfreeze_blocks"])
                optimizer = torch.optim.AdamW(optimizer_groups(model, cfg, reductions))
                row["action"] = "unfreeze_encoder_tail"
            elif phase == "encoder_tail" and stale > 0 and stale % cfg["phase_b_lr_patience"] == 0 \
                    and reductions < cfg["phase_b_max_lr_reductions"]:
                reductions += 1
                optimizer = torch.optim.AdamW(optimizer_groups(model, cfg, reductions))
                row["action"] = "halve_learning_rates"
            history.append(row)
            append_jsonl(run_dir / "development_curve.jsonl", row)
            torch.save({
                "contract": contract, "epoch": epoch, "phase": phase, "stale": stale,
                "lr_reductions": reductions, "best_epoch": best_epoch, "best_key": list(best_key),
                "model": model.state_dict(), "optimizer": optimizer.state_dict(), "history": history,
            }, run_dir / "development_latest.tmp")
            os.replace(run_dir / "development_latest.tmp", run_dir / "development_latest.pt")
            if epoch % cfg["sparse_checkpoint_every_epochs"] == 0:
                torch.save({"contract": contract, "epoch": epoch, "model": model.state_dict()},
                           run_dir / f"development_epoch{epoch:03d}.pt")
            atomic_json(report_dir / "status.json", {
                "status": "development_training", "epoch": epoch, "phase": phase,
                "latest_validation": validation,
                "best_epoch": best_epoch, "best_acc_all": best_key[0], "stale": stale,
                "lr_reductions": reductions, "microbatch": microbatch, "test_forward_count": 0,
            })
            if phase == "encoder_tail" and stale >= cfg["phase_b_stop_patience"] \
                    and reductions >= cfg["phase_b_max_lr_reductions"]:
                break
        if STOP:
            atomic_json(report_dir / "status.json", {
                "status": "paused", "epoch": history[-1]["epoch"] if history else 0,
                "best_epoch": best_epoch, "test_forward_count": 0,
            })
            return 0

        # Paper endpoint: fresh model, all 30 source-train images per class, and
        # the exact phase/LR schedule used through the selected development epoch.
        selected_schedule = history[:best_epoch]
        model, _ = build_model(cfg, device)
        current_phase = None
        current_reductions = None
        optimizer = None
        refit_rng = np.random.default_rng(cfg["seed"] + 9302)
        atomic_json(report_dir / "status.json", {
            "status": "paper_refit", "epoch": 0, "selected_epochs": best_epoch,
            "test_forward_count": 0,
        })
        for schedule in selected_schedule:
            scheduled_phase = schedule["phase"]
            scheduled_reductions = int(schedule["lr_reductions"])
            if scheduled_phase != current_phase or scheduled_reductions != current_reductions:
                current_phase = scheduled_phase
                current_reductions = scheduled_reductions
                set_phase(model, current_phase, cfg["phase_b_unfreeze_blocks"])
                optimizer = torch.optim.AdamW(optimizer_groups(model, cfg, scheduled_reductions))
            train = train_epoch(
                model, waveform, labels, data["official_train"], frequency_mean, frequency_scale,
                optimizer, cfg, microbatch, device, refit_rng,
            )
            row = {
                "stage": "paper_refit", "epoch": schedule["epoch"], "phase": current_phase,
                "train": train, "replayed_lr_reductions": scheduled_reductions,
                "test_forward_count": 0,
            }
            append_jsonl(run_dir / "paper_refit_curve.jsonl", row)
            atomic_json(report_dir / "status.json", {
                "status": "paper_refit", "epoch": schedule["epoch"], "selected_epochs": best_epoch,
                "phase": current_phase, "train": train, "test_forward_count": 0,
            })
        torch.save({
            "contract": contract, "selected_development_epoch": best_epoch,
            "model": model.state_dict(), "schedule": selected_schedule,
        }, run_dir / "paper_refit.pt")

        test, predictions = evaluate(
            model, waveform, labels, data["test"], frequency_mean, frequency_scale,
            microbatch, device, predictions=True,
        )
        for row, index in zip(predictions, data["test"].tolist()):
            row["trial_id"] = str(data["trial_ids"][index])
            row["image_id"] = str(data["images"][index])
            row["source_index"] = int(data["source_indices"][index])
        with (run_dir / "paper_test_predictions.jsonl").open("w", encoding="utf-8") as stream:
            for row in predictions:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        result = {
            "status": "complete", "subject": cfg["subject"], "stage": 1,
            "development_best": torch.load(best_path, map_location="cpu", weights_only=False)["validation"],
            "selected_development_epoch": best_epoch,
            "paper_refit_train_count": len(data["official_train"]),
            "paper_test": test,
            "test_forward_count": 1,
            "parameters": parameter_counts(model),
            "checkpoints": {
                "development_best": str(best_path), "paper_refit": str(run_dir / "paper_refit.pt"),
            },
            "contract": contract,
        }
        atomic_json(run_dir / "result.json", result)
        atomic_json(report_dir / "status.json", result)
        return 0
    finally:
        stopped.set()
        monitor.join(timeout=2)


if __name__ == "__main__":
    for current_signal in (signal.SIGINT, signal.SIGTERM):
        signal.signal(current_signal, request_stop)
    try:
        raise SystemExit(main())
    except Exception as error:
        cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
        atomic_json(ROOT / cfg["report_dir"] / "status.json", {
            "status": "failed", "error_type": type(error).__name__, "error": str(error),
            "test_forward_count": 0,
        })
        raise
