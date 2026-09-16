"""Local D089 masked-waveform pretraining with validation-curve stopping."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from d089_masked_retrieval import (  # noqa: E402
    CHANNELS,
    D089Encoder,
    MaskedWaveformReconstructor,
    parameter_counts,
    periodic_hann_log_psd,
)
import run_d089_corrected_masked_retrieval as common  # noqa: E402


CONFIG = ROOT / "config/d089_local_curve_pretrain.json"
STOP = False


def request_stop(*_args) -> None:
    global STOP
    STOP = True


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def optimizer_groups(model: torch.nn.Module, weight_decay: float):
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim == 1 or name.endswith(".bias") or "norm" in name.lower():
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def cpu_statistics(waveform: torch.Tensor, fit: torch.Tensor):
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
    fit_waveform = ((waveform[fit] - mean.float()[None, :, None]) / scale.float()[None, :, None]).contiguous()
    return mean.float(), scale.float(), fit_waveform


def frequency_statistics(fit_waveform: torch.Tensor):
    total = torch.zeros((CHANNELS, 32), dtype=torch.float64)
    square = torch.zeros((CHANNELS, 32), dtype=torch.float64)
    count = 0
    for value in fit_waveform.split(32):
        frequency = periodic_hann_log_psd(value).double()
        total += frequency.sum(0)
        square += frequency.square().sum(0)
        count += len(value)
    mean = total / count
    scale = (square / count - mean.square()).clamp_min(1e-12).sqrt()
    return mean.float(), scale.float()


def pin(value: torch.Tensor) -> tuple[torch.Tensor, bool]:
    try:
        return value.pin_memory(), True
    except RuntimeError:
        return value, False


def capacity_probe(cfg, waveform, frequency_mean, frequency_scale, device):
    attempts = []
    chosen = None
    for candidate in cfg["microbatch_candidates"]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        try:
            torch.manual_seed(cfg["seed"])
            model = MaskedWaveformReconstructor(
                D089Encoder(
                    "separate_conv",
                    dropout=cfg["dropout"],
                    activation_checkpointing=True,
                    frequency_fusion_scale=cfg["frequency_fusion_scale"],
                )
            ).to(device)
            optimizer = torch.optim.AdamW(optimizer_groups(model, cfg["weight_decay"]), lr=cfg["peak_lr"])
            batch = waveform[:candidate].to(device, non_blocking=True)
            hidden = common.make_masks(
                candidate, cfg["validation_mask_ratio"],
                torch.Generator().manual_seed(cfg["seed"] + 9100 + candidate), device,
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, _ = model.loss(batch, hidden, frequency_mean, frequency_scale)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"])
            optimizer.step()
            torch.cuda.synchronize(device)
            peak = torch.cuda.max_memory_allocated(device) / 2**30
            attempts.append({"microbatch": candidate, "status": "PASS", "peak_allocated_gib": peak})
            if peak <= cfg["peak_memory_limit_gib"]:
                chosen = candidate
                del model, optimizer, batch, hidden, loss
                torch.cuda.empty_cache()
                break
            del model, optimizer, batch, hidden, loss
        except torch.cuda.OutOfMemoryError:
            attempts.append({"microbatch": candidate, "status": "OOM"})
        finally:
            torch.cuda.empty_cache()
    if chosen is None:
        raise RuntimeError(f"No local microbatch passed capacity probe: {attempts}")
    return chosen, attempts


@torch.inference_mode()
def validate(model, waveform, masks, frequency_mean, frequency_scale, microbatch, device):
    model.eval()
    totals = {"loss": 0.0, "wave": 0.0, "slope": 0.0}
    seen = 0
    for start in range(0, len(waveform), microbatch):
        stop = min(start + microbatch, len(waveform))
        batch = waveform[start:stop].to(device, non_blocking=True)
        hidden = masks[start:stop].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, parts = model.loss(batch, hidden, frequency_mean, frequency_scale)
        count = len(batch)
        totals["loss"] += float(loss) * count
        totals["wave"] += float(parts["wave"]) * count
        totals["slope"] += float(parts["slope"]) * count
        seen += count
    return {key: value / seen for key, value in totals.items()}


def monitor_gpu(cfg, output: Path, stop_event: threading.Event):
    output.parent.mkdir(parents=True, exist_ok=True)
    if not output.exists():
        output.write_text(
            "timestamp_unix,gpu_uuid,utilization_gpu_percent,memory_used_mib,memory_total_mib,temperature_c\n",
            encoding="utf-8",
        )
    while not stop_event.is_set():
        result = subprocess.run(
            ["nvidia-smi", f"--id={cfg['gpu_uuid']}",
             "--query-gpu=uuid,utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            with output.open("a", encoding="utf-8") as stream:
                stream.write(f"{time.time()},{result.stdout.strip()}\n")
        stop_event.wait(30)


def checkpoint_contract(cfg, route, split):
    return {
        "task": cfg["task"], "route": route,
        "config_sha256": file_sha256(CONFIG),
        "model_sha256": file_sha256(ROOT / "src/d089_masked_retrieval.py"),
        "runner_sha256": file_sha256(Path(__file__)),
        "fit_hash": tensor_sha256(split["fit"]),
        "validation_hash": tensor_sha256(split["validation"]),
    }


def save_resume(path, model, optimizer, route, step, history, state, contract, rng, mask_rng, cycle):
    common.save_checkpoint(path, {
        "kind": "d089_masked_waveform_local", "route": route, "step": step,
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "history": history, "curve_state": state, "contract": contract,
        "numpy_rng": rng.bit_generator.state, "mask_rng": mask_rng.get_state(),
        "sampler_cycle": cycle,
        "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
    })


def train_route(cfg, route, fit_waveform, validation_waveform, validation_masks,
                frequency_mean, frequency_scale, microbatch, split, device):
    run = ROOT / cfg["run_dir"] / route
    run.mkdir(parents=True, exist_ok=True)
    result_path = run / "result.json"
    contract = checkpoint_contract(cfg, route, split)
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result["contract"] != contract:
            raise RuntimeError(f"Completed route contract changed: {route}")
        return result

    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])
    model = MaskedWaveformReconstructor(
        D089Encoder(
            route,
            dropout=cfg["dropout"],
            activation_checkpointing=True,
            frequency_fusion_scale=cfg["frequency_fusion_scale"],
        )
    ).to(device)
    optimizer = torch.optim.AdamW(optimizer_groups(model, cfg["weight_decay"]), lr=cfg["peak_lr"])
    rng = np.random.default_rng(cfg["seed"] + 9200)
    mask_rng = torch.Generator().manual_seed(cfg["seed"] + 9201)
    cycle = {}
    history = []
    curve = {
        "mask_ratio": cfg["mask_start"], "stage_best": float("inf"),
        "absolute_best": float("inf"), "stage_stale": 0, "full_mask_stale": 0,
        "lr_reductions": 0, "current_lr": cfg["peak_lr"], "best_step": 0,
    }
    start_step = 0
    latest = run / "latest.pt"
    if latest.exists():
        saved = torch.load(latest, map_location=device, weights_only=False)
        if saved["contract"] != contract:
            raise RuntimeError(f"Resume contract changed: {route}")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        history, curve, start_step = saved["history"], saved["curve_state"], int(saved["step"])
        rng.bit_generator.state = saved["numpy_rng"]
        mask_rng.set_state(saved["mask_rng"].cpu())
        cycle = saved["sampler_cycle"]
        torch.set_rng_state(saved["torch_rng"].cpu())
        torch.cuda.set_rng_state_all([value.cpu() for value in saved["cuda_rng"]])

    logical_indices = torch.arange(len(fit_waveform))
    interval = {"loss": 0.0, "wave": 0.0, "slope": 0.0, "seen": 0}
    started = time.perf_counter()
    stop_reason = "safety_cap"
    for step in range(start_step + 1, cfg["max_steps_safety_cap"] + 1):
        if STOP:
            stop_reason = "signal"
            save_resume(latest, model, optimizer, route, step - 1, history, curve, contract, rng, mask_rng, cycle)
            break
        logical = common.shuffled_cycle(logical_indices, cfg["logical_batch"], rng, cycle)
        used_mask_ratio = curve["mask_ratio"]
        masks = common.make_masks(len(logical), used_mask_ratio, mask_rng, device)
        if step <= cfg["warmup_steps"] and curve["lr_reductions"] == 0:
            lr = cfg["peak_lr"] * step / cfg["warmup_steps"]
        else:
            lr = curve["current_lr"]
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        for start in range(0, len(logical), microbatch):
            ids = logical[start:start + microbatch]
            batch = fit_waveform[ids].to(device, non_blocking=True)
            hidden = masks[start:start + microbatch]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, parts = model.loss(batch, hidden, frequency_mean, frequency_scale)
            weight = len(batch) / len(logical)
            (loss * weight).backward()
            interval["loss"] += float(loss.detach()) * len(batch)
            interval["wave"] += float(parts["wave"]) * len(batch)
            interval["slope"] += float(parts["slope"]) * len(batch)
            interval["seen"] += len(batch)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"]))
        optimizer.step()

        if step % cfg["validate_every_steps"] != 0:
            continue
        metrics = validate(model, validation_waveform, validation_masks, frequency_mean,
                           frequency_scale, microbatch, device)
        raw_best = metrics["loss"] < curve["absolute_best"]
        material = metrics["loss"] < curve["stage_best"] * (1.0 - cfg["relative_improvement"])
        if raw_best:
            curve["absolute_best"] = metrics["loss"]
            curve["best_step"] = step
            common.save_checkpoint(run / "best.pt", {
                "kind": "d089_masked_waveform_best", "route": route, "step": step,
                "model": model.state_dict(), "validation": metrics, "contract": contract,
                "mask_ratio": curve["mask_ratio"], "parameters": parameter_counts(model),
            })
        if material:
            curve["stage_best"] = metrics["loss"]
            curve["stage_stale"] = 0
            curve["full_mask_stale"] = 0
        else:
            curve["stage_stale"] += 1
            if curve["mask_ratio"] >= cfg["mask_max"] - 1e-9:
                curve["full_mask_stale"] += 1

        action = "continue"
        if curve["mask_ratio"] < cfg["mask_max"] - 1e-9 \
                and curve["stage_stale"] >= cfg["mask_stage_patience_validations"]:
            curve["mask_ratio"] = min(cfg["mask_max"], curve["mask_ratio"] + cfg["mask_increment"])
            curve["stage_best"] = metrics["loss"]
            curve["stage_stale"] = 0
            action = "increase_mask"
            common.save_checkpoint(run / f"mask{int(round(curve['mask_ratio'] * 100)):02d}_entry.pt", {
                "kind": "d089_mask_stage_entry", "route": route, "step": step,
                "model": model.state_dict(), "validation": metrics, "contract": contract,
            })
        elif curve["mask_ratio"] >= cfg["mask_max"] - 1e-9 \
                and curve["full_mask_stale"] >= cfg["lr_patience_validations"] \
                and curve["lr_reductions"] < cfg["max_lr_reductions"]:
            curve["current_lr"] = max(cfg["minimum_lr"], curve["current_lr"] * 0.5)
            curve["lr_reductions"] += 1
            curve["full_mask_stale"] = 0
            curve["stage_best"] = metrics["loss"]
            action = "reduce_lr"
        elif curve["mask_ratio"] >= cfg["mask_max"] - 1e-9 \
                and curve["lr_reductions"] >= cfg["max_lr_reductions"] \
                and curve["full_mask_stale"] >= cfg["stop_patience_validations"]:
            action = "stop_plateau"
            stop_reason = "validation_plateau_after_two_lr_reductions"

        row = {
            "step": step, "route": route, "train_mask_ratio": used_mask_ratio,
            "train": {key: interval[key] / interval["seen"] for key in ("loss", "wave", "slope")},
            "validation_fixed_mask50": metrics, "lr": lr, "grad_norm": grad_norm,
            "action": action, "curve_state": dict(curve),
            "elapsed_seconds": time.perf_counter() - started,
            "allocated_gib": torch.cuda.memory_allocated() / 2**30,
        }
        history.append(row)
        common.append_jsonl(run / "curve.jsonl", row)
        common.atomic_json(ROOT / cfg["report_dir"] / "status.json", {
            "status": "running", "route": route, **row,
        })
        print(json.dumps(row), flush=True)
        interval = {"loss": 0.0, "wave": 0.0, "slope": 0.0, "seen": 0}
        save_resume(latest, model, optimizer, route, step, history, curve, contract, rng, mask_rng, cycle)
        if action == "stop_plateau":
            break

    result = {
        "status": "complete" if stop_reason != "signal" else "paused",
        "route": route, "stop_reason": stop_reason, "last_step": step,
        "best_step": curve["best_step"], "best_validation_loss": curve["absolute_best"],
        "microbatch": microbatch, "logical_batch": cfg["logical_batch"],
        "history_points": len(history), "contract": contract,
    }
    if result["status"] == "complete":
        common.atomic_json(result_path, result)
    del model, optimizer
    torch.cuda.empty_cache()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        parser.error("--run is required")
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    if not torch.cuda.is_available():
        raise RuntimeError("Local D089 pretraining requires CUDA")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != cfg["gpu_uuid"]:
        raise RuntimeError("Local D089 must run inside its UUID lock")
    device = torch.device("cuda")
    data = common.load_subject_data(cfg)
    split = {"fit": data["fit"], "validation": data["validation"]}
    mean, scale, fit_waveform = cpu_statistics(data["waveform"], data["fit"])
    validation_waveform = (
        (data["waveform"][data["validation"]] - mean[None, :, None]) / scale[None, :, None]
    ).contiguous()
    del data["waveform"]
    frequency_mean, frequency_scale = frequency_statistics(fit_waveform)
    fit_waveform, fit_pinned = pin(fit_waveform)
    validation_waveform, validation_pinned = pin(validation_waveform)
    validation_masks = common.make_masks(
        len(validation_waveform), cfg["validation_mask_ratio"],
        torch.Generator().manual_seed(cfg["validation_mask_seed"]), torch.device("cpu"),
    )
    validation_masks, masks_pinned = pin(validation_masks)
    frequency_mean = frequency_mean.to(device)
    frequency_scale = frequency_scale.to(device)
    microbatch, attempts = capacity_probe(cfg, fit_waveform, frequency_mean, frequency_scale, device)
    report = ROOT / cfg["report_dir"]
    report.mkdir(parents=True, exist_ok=True)
    common.atomic_json(report / "preflight.json", {
        "status": "PASS", "gpu_uuid": cfg["gpu_uuid"], "microbatch": microbatch,
        "capacity_attempts": attempts, "logical_batch": cfg["logical_batch"],
        "fit": len(fit_waveform), "validation": len(validation_waveform),
        "fit_per_class": 27, "validation_per_class": 3,
        "test_forward_count": 0, "wave_mean_hash": tensor_sha256(mean),
        "wave_scale_hash": tensor_sha256(scale), "frequency_mean_hash": tensor_sha256(frequency_mean),
        "frequency_scale_hash": tensor_sha256(frequency_scale),
        "pinned": {"fit": fit_pinned, "validation": validation_pinned, "masks": masks_pinned},
        "regularization": {"dropout": cfg["dropout"], "weight_decay": cfg["weight_decay"],
                           "gradient_clip": cfg["gradient_clip"]},
        "frequency_fusion_scale": cfg["frequency_fusion_scale"],
    })
    stop_event = threading.Event()
    monitor = threading.Thread(
        target=monitor_gpu, args=(cfg, report / "gpu_samples.csv", stop_event), daemon=True
    )
    monitor.start()
    results = []
    try:
        for route in cfg["routes"]:
            if STOP:
                break
            result = train_route(
                cfg, route, fit_waveform, validation_waveform, validation_masks,
                frequency_mean, frequency_scale, microbatch, split, device,
            )
            results.append(result)
            common.atomic_json(report / "routes.json", results)
    finally:
        stop_event.set()
        monitor.join(timeout=5)
    complete = len(results) == len(cfg["routes"]) and all(row["status"] == "complete" for row in results)
    common.atomic_json(report / "status.json", {
        "status": "complete" if complete else "paused", "routes": results,
    })
    return 0 if complete or STOP else 1


if __name__ == "__main__":
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, request_stop)
    raise SystemExit(main())
