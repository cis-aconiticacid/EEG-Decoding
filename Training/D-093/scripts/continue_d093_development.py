"""Continue the underfit D093 development run without touching test EEG."""

from __future__ import annotations

import json
import os
import signal
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_d089_corrected_masked_retrieval as common  # noqa: E402
import run_d093_stage1_classification_baseline as d093  # noqa: E402


STOP = False


def request_stop(*_args) -> None:
    global STOP
    STOP = True


def main() -> int:
    cfg = json.loads(d093.CONFIG.read_text(encoding="utf-8"))
    if os.environ.get("CUDA_VISIBLE_DEVICES") != cfg["gpu_uuid"] or not torch.cuda.is_available():
        raise RuntimeError("D093 continuation must use the configured local GPU UUID")
    device = torch.device("cuda")
    report_dir = ROOT / cfg["report_dir"]
    run_dir = ROOT / cfg["run_dir"]
    source = run_dir / "development_latest.pt"
    saved = torch.load(source, map_location=device, weights_only=False)

    data = common.load_subject_data(cfg)
    wave_mean, wave_scale, waveform = d093.waveform_statistics(data["waveform"], data["fit"])
    frequency_mean, frequency_scale = d093.frequency_statistics(waveform, data["fit"])
    try:
        waveform = waveform.pin_memory()
    except RuntimeError:
        pass
    frequency_mean = frequency_mean.to(device)
    frequency_scale = frequency_scale.to(device)
    labels = data["labels"]

    model, _ = d093.build_model(cfg, device)
    phase = "encoder_tail"
    d093.set_phase(model, phase, cfg["phase_b_unfreeze_blocks"])
    model.load_state_dict(saved["model"], strict=True)
    optimizer = torch.optim.AdamW(d093.optimizer_groups(model, cfg, reductions=0))
    try:
        optimizer.load_state_dict(saved["optimizer"])
    except (ValueError, KeyError):
        pass
    # The preceding stop was caused by an underfit validation plateau. Restore
    # the planned base rates while preserving optimizer moments when compatible.
    for group in optimizer.param_groups:
        group["lr"] = cfg["encoder_lr"] if group["role"] == "encoder" else cfg["head_lr"]

    start_epoch = int(saved["epoch"])
    best_epoch = int(saved["best_epoch"])
    best_key = tuple(saved["best_key"])
    max_epoch = 150
    reductions = 0
    val_stale = 0
    train_history: list[float] = []
    rng = np.random.default_rng(cfg["seed"] + 9350 + start_epoch)
    microbatch = 64
    started = time.perf_counter()
    stopped = threading.Event()
    monitor = threading.Thread(
        target=d093.monitor_gpu,
        args=(cfg, report_dir / "continuation_gpu_samples.csv", stopped),
        daemon=True,
    )
    monitor.start()
    continuation_contract = {
        "source_checkpoint": str(source),
        "source_epoch": start_epoch,
        "source_contract": saved["contract"],
        "restored_learning_rates": {"encoder": cfg["encoder_lr"], "head": cfg["head_lr"]},
        "underfit_gate": 0.50,
        "joint_plateau_window": 10,
        "validation_patience_after_gate": 15,
        "max_epoch": max_epoch,
        "test_forward_count_during_continuation": 0,
        "premature_prior_test_is_not_used_for_selection": True,
    }
    d093.atomic_json(report_dir / "continuation_preflight.json", continuation_contract)
    d093.atomic_json(report_dir / "status.json", {
        "status": "continuation_training", "epoch": start_epoch, "phase": phase,
        "best_epoch": best_epoch, "best_acc_all": best_key[0],
        "test_forward_count_during_continuation": 0,
    })

    try:
        for epoch in range(start_epoch + 1, max_epoch + 1):
            if STOP:
                break
            train = d093.train_epoch(
                model, waveform, labels, data["fit"], frequency_mean, frequency_scale,
                optimizer, cfg, microbatch, device, rng,
            )
            validation, _ = d093.evaluate(
                model, waveform, labels, data["validation"], frequency_mean, frequency_scale,
                microbatch, device,
            )
            key = (validation["acc_all"], -validation["loss"])
            improved = key > best_key
            if improved:
                best_key = key
                best_epoch = epoch
                val_stale = 0
                torch.save({
                    "contract": saved["contract"], "continuation_contract": continuation_contract,
                    "epoch": epoch, "phase": phase, "model": model.state_dict(),
                    "validation": validation, "selection_key": list(key),
                }, run_dir / "development_best.pt")
            else:
                val_stale += 1
            train_history.append(float(train["acc_all"]))
            train_plateau = False
            if len(train_history) >= 10:
                previous = max(train_history[-10:-5])
                recent = max(train_history[-5:])
                train_plateau = recent - previous < 0.01

            action = None
            gate_open = train["acc_all"] >= 0.50
            if gate_open and train_plateau and val_stale >= 15 and reductions < 2:
                reductions += 1
                for group in optimizer.param_groups:
                    base = cfg["encoder_lr"] if group["role"] == "encoder" else cfg["head_lr"]
                    group["lr"] = base * (0.5 ** reductions)
                val_stale = 0
                action = "joint_plateau_halve_learning_rates"

            row = {
                "stage": "development_continuation", "epoch": epoch, "phase": phase,
                "train": train, "validation": validation, "selection_key": list(key),
                "improved": improved, "best_epoch": best_epoch, "best_acc_all": best_key[0],
                "validation_stale": val_stale, "train_plateau": train_plateau,
                "underfit_gate_open": gate_open, "lr_reductions": reductions,
                "learning_rates": {group["role"]: group["lr"] for group in optimizer.param_groups},
                "action": action, "elapsed_seconds": time.perf_counter() - started,
                "test_forward_count_during_continuation": 0,
            }
            d093.append_jsonl(run_dir / "continuation_curve.jsonl", row)
            state = {
                "contract": saved["contract"], "continuation_contract": continuation_contract,
                "epoch": epoch, "phase": phase, "best_epoch": best_epoch,
                "best_key": list(best_key), "val_stale": val_stale,
                "lr_reductions": reductions, "train_history": train_history,
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            }
            torch.save(state, run_dir / "continuation_latest.tmp")
            os.replace(run_dir / "continuation_latest.tmp", run_dir / "continuation_latest.pt")
            if epoch % cfg["sparse_checkpoint_every_epochs"] == 0:
                torch.save(state, run_dir / f"continuation_epoch{epoch:03d}.pt")
            d093.atomic_json(report_dir / "status.json", {
                "status": "continuation_training", "epoch": epoch, "phase": phase,
                "latest_train": train, "latest_validation": validation,
                "best_epoch": best_epoch, "best_acc_all": best_key[0],
                "underfit_gate_open": gate_open, "train_plateau": train_plateau,
                "validation_stale": val_stale, "lr_reductions": reductions,
                "test_forward_count_during_continuation": 0,
            })

            reached_target = validation["acc_all"] >= 0.90
            joint_plateau = gate_open and train_plateau and reductions >= 2 and val_stale >= 15
            if reached_target or joint_plateau:
                reason = "validation_reached_90_percent" if reached_target else "joint_train_validation_plateau"
                d093.atomic_json(report_dir / "status.json", {
                    "status": "awaiting_review", "reason": reason, "epoch": epoch,
                    "latest_train": train, "latest_validation": validation,
                    "best_epoch": best_epoch, "best_acc_all": best_key[0],
                    "test_forward_count_during_continuation": 0,
                })
                return 0

        final_epoch = epoch if "epoch" in locals() else start_epoch
        d093.atomic_json(report_dir / "status.json", {
            "status": "paused" if STOP else "awaiting_review",
            "reason": "external_stop" if STOP else "safety_cap_150",
            "epoch": final_epoch, "best_epoch": best_epoch, "best_acc_all": best_key[0],
            "test_forward_count_during_continuation": 0,
        })
        return 0
    finally:
        stopped.set()
        monitor.join(timeout=2)


if __name__ == "__main__":
    for current_signal in (signal.SIGINT, signal.SIGTERM):
        signal.signal(current_signal, request_stop)
    raise SystemExit(main())
