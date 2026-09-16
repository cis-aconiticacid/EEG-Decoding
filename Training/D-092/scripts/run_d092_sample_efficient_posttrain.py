"""Subject-0 sample-efficient posttraining into the full RAEv2 spatial latent."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_d089_corrected_masked_retrieval as common  # noqa: E402
from d089_masked_retrieval import (  # noqa: E402
    CHANNELS,
    IMAGE_DIM,
    IMAGE_TOKENS,
    PATCHES,
    D089Encoder,
    normalized_flat,
    periodic_hann_log_psd,
)
from d092_sample_efficient_latent import (  # noqa: E402
    SampleEfficientLatentPredictor,
    latent_objective,
    trainable_parameter_counts,
)


CONFIG = ROOT / "config/d092_sample_efficient_posttrain.json"
STOP = False


def request_stop(*_args) -> None:
    global STOP
    STOP = True


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
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
        frequency = periodic_hann_log_psd(standardized[indices]).double()
        total += frequency.sum(0)
        square += frequency.square().sum(0)
        count += len(indices)
    mean = total / count
    scale = (square / count - mean.square()).clamp_min(1e-12).sqrt()
    return mean.float(), scale.float()


def load_targets(cfg: dict, data: dict):
    target_dir = ROOT / cfg["target_dir"]
    contract_path = target_dir / "targets.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract["shape_per_image"] != [IMAGE_TOKENS, IMAGE_DIM] or contract["all_image_count"] != 4000:
        raise RuntimeError("D091 target shape/count contract failed")
    targets = torch.empty((4000, IMAGE_TOKENS, IMAGE_DIM), dtype=torch.float32)
    records = []
    offset = 0
    for split in ("train", "monitor_val", "final_holdout"):
        meta = contract["splits"][split]
        path = target_dir / f"{split}.npy"
        if not meta["complete"] or sha256(path) != meta["array_sha256"]:
            raise RuntimeError(f"target hash failed: {split}")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if tuple(array.shape[1:]) != (IMAGE_TOKENS, IMAGE_DIM) or array.dtype != np.float32:
            raise RuntimeError(f"target array contract failed: {split}")
        for start in range(0, len(array), 32):
            stop = min(start + 32, len(array))
            targets[offset + start:offset + stop].copy_(torch.from_numpy(np.asarray(array[start:stop]).copy()))
        records.extend(meta["images"])
        offset += len(array)
    lookup = {record["image_id"]: index for index, record in enumerate(records)}
    if offset != 4000 or len(lookup) != 4000 or not torch.isfinite(targets).all():
        raise RuntimeError("target residency validation failed")
    trial_targets = torch.tensor([lookup[str(image)] for image in data["images"]], dtype=torch.long)
    gallery_labels = torch.tensor([int(record["label_index"]) for record in records], dtype=torch.long)
    if not torch.equal(gallery_labels[trial_targets], data["labels"]):
        raise RuntimeError("EEG/image target label mapping failed")
    return targets, trial_targets, gallery_labels, contract, contract_path


def encoder_from_best(cfg: dict) -> tuple[D089Encoder, dict]:
    path = ROOT / cfg["pretrain_best"]
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if int(saved.get("step", -1)) != cfg["pretrain_expected_step"] or saved.get("route") != cfg["route"]:
        raise RuntimeError("pretraining best checkpoint identity failed")
    state = saved["model"]
    encoder_state = {name.removeprefix("encoder."): value for name, value in state.items() if name.startswith("encoder.")}
    encoder = D089Encoder(
        cfg["route"], dropout=cfg["dropout"], activation_checkpointing=True, frequency_fusion_scale=0.1
    )
    encoder.load_state_dict(encoder_state, strict=True)
    return encoder, saved


def set_phase(model: SampleEfficientLatentPredictor, phase: str, unfreeze_blocks: int) -> None:
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(False)
    if phase == "encoder_tail":
        for block in model.encoder.blocks[-unfreeze_blocks:]:
            for parameter in block.parameters():
                parameter.requires_grad_(True)
        for module in (model.encoder.final_norm, model.encoder.frequency_q_norm, model.encoder.frequency_kv_norm):
            for parameter in module.parameters():
                parameter.requires_grad_(True)
    for name, parameter in model.named_parameters():
        if not name.startswith("encoder."):
            parameter.requires_grad_(True)


def optimizer_groups(model: SampleEfficientLatentPredictor, cfg: dict, phase: str):
    groups = []
    for prefix, lr in (("encoder.", cfg["encoder_lr"]), ("", cfg["head_peak_lr"])):
        decay, no_decay = [], []
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            is_encoder = name.startswith("encoder.")
            if (prefix == "encoder.") != is_encoder:
                continue
            target = no_decay if parameter.ndim == 1 or name.endswith(".bias") or "norm" in name.lower() else decay
            target.append(parameter)
        if decay:
            groups.append({"params": decay, "lr": lr, "weight_decay": cfg["weight_decay"], "role": prefix or "head"})
        if no_decay:
            groups.append({"params": no_decay, "lr": lr, "weight_decay": 0.0, "role": prefix or "head"})
    return groups


def augment_view(waveform: torch.Tensor, cfg: dict) -> tuple[torch.Tensor, torch.Tensor]:
    settings = cfg["augmentation"]
    batch = len(waveform)
    value = waveform.clone()
    gain = torch.empty((batch, 1, 1), device=value.device).uniform_(settings["gain_min"], settings["gain_max"])
    value *= gain
    noise_scale = torch.empty((batch, 1, 1), device=value.device).uniform_(0.0, settings["noise_std_max"])
    value += torch.randn_like(value) * noise_scale
    keep = torch.rand((batch, CHANNELS, 1), device=value.device) >= settings["channel_dropout"]
    value *= keep
    max_shift = int(settings["max_shift_samples"])
    shifts = torch.randint(-max_shift, max_shift + 1, (batch,), device=value.device)
    for row, shift_tensor in enumerate(shifts):
        shift = int(shift_tensor)
        if shift:
            value[row] = torch.roll(value[row], shift, dims=-1)
            if shift > 0:
                value[row, :, :shift] = 0
            else:
                value[row, :, shift:] = 0
    hidden = torch.zeros((batch, PATCHES), dtype=torch.bool, device=value.device)
    selected = torch.rand(batch, device=value.device) < settings["mask_one_patch_probability"]
    starts = torch.randint(PATCHES, (batch,), device=value.device)
    hidden[torch.arange(batch, device=value.device)[selected], starts[selected]] = True
    return value, hidden


@torch.inference_mode()
def predict(model, waveform, indices, frequency_mean, frequency_scale, microbatch, device):
    model.eval()
    outputs = []
    for ids in indices.split(microbatch):
        batch = waveform[ids].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(batch, frequency_mean, frequency_scale)
        outputs.append(output.float().cpu())
    return torch.cat(outputs)


@torch.inference_mode()
def retrieval_metrics(predicted, true_target_indices, true_labels, candidate_indices, targets, gallery_labels, device):
    predicted_flat = normalized_flat(predicted.to(device))
    count = len(predicted)
    top_scores = torch.full((count, 5), -torch.inf, device=device)
    top_indices = torch.full((count, 5), -1, dtype=torch.long, device=device)
    class_scores = torch.full((count, 80), -torch.inf, device=device)
    for candidate_chunk in candidate_indices.split(16):
        gallery = normalized_flat(targets[candidate_chunk].to(device))
        scores = predicted_flat @ gallery.T
        merged_scores = torch.cat((top_scores, scores), dim=1)
        merged_indices = torch.cat((top_indices, candidate_chunk.to(device)[None].expand(count, -1)), dim=1)
        top_scores, positions = merged_scores.topk(5, dim=1)
        top_indices = merged_indices.gather(1, positions)
        labels = gallery_labels[candidate_chunk].to(device)
        for label in labels.unique():
            local = labels == label
            class_scores[:, int(label)] = torch.maximum(class_scores[:, int(label)], scores[:, local].amax(1))
    true_targets = true_target_indices.to(device)
    labels = true_labels.to(device)
    exact_top1 = float((top_indices[:, 0] == true_targets).float().mean())
    exact_top5 = float((top_indices == true_targets[:, None]).any(1).float().mean())
    predicted_labels = class_scores.argmax(1)
    category_accuracy = float((predicted_labels == labels).float().mean())
    regression_sum = cosine_sum = 0.0
    for start in range(0, count, 8):
        stop = min(start + 8, count)
        estimate = predicted[start:stop].to(device)
        actual = targets[true_target_indices[start:stop]].to(device)
        regression_sum += float(F.smooth_l1_loss(estimate, actual, beta=1.0, reduction="sum"))
        cosine_sum += float(F.cosine_similarity(estimate, actual, dim=-1).sum())
    regression = regression_sum / (count * IMAGE_TOKENS * IMAGE_DIM)
    token_cosine = cosine_sum / (count * IMAGE_TOKENS)
    return {
        "candidate_images": len(candidate_indices),
        "exact_image_top1": exact_top1,
        "exact_image_top5": exact_top5,
        "category_accuracy": category_accuracy,
        "smooth_l1": regression,
        "token_cosine": token_cosine,
        "random": {"exact_top1": 1 / len(candidate_indices), "category_top1": 1 / 80},
    }


def selection_key(metrics: dict) -> tuple[float, float, float, float]:
    return (
        metrics["category_accuracy"], metrics["exact_image_top5"],
        metrics["token_cosine"], -metrics["smooth_l1"],
    )


def capacity_probe(cfg, encoder_state, waveform, targets, trial_targets, frequency_mean, frequency_scale, device):
    attempts = []
    for candidate in cfg["microbatch_candidates"]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            encoder = D089Encoder(cfg["route"], dropout=cfg["dropout"], activation_checkpointing=True,
                                  frequency_fusion_scale=0.1)
            encoder.load_state_dict(encoder_state, strict=True)
            model = SampleEfficientLatentPredictor(encoder, cfg["query_depth"], cfg["dropout"]).to(device)
            set_phase(model, "encoder_tail", cfg["phase_b_unfreeze_blocks"])
            optimizer = torch.optim.AdamW(optimizer_groups(model, cfg, "encoder_tail"))
            ids = torch.arange(candidate)
            batch = waveform[ids].to(device)
            target = targets[trial_targets[ids]].to(device)
            negatives = targets[trial_targets[torch.arange(candidate, candidate + cfg["negative_targets"])]] .to(device)
            first, first_mask = augment_view(batch, cfg)
            second, second_mask = augment_view(batch, cfg)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                first_prediction = model(first, frequency_mean, frequency_scale, first_mask)
                second_prediction = model(second, frequency_mean, frequency_scale, second_mask)
                loss, _ = latent_objective(
                    first_prediction, second_prediction, target, negatives, cfg["temperature"],
                    cfg["cosine_weight"], cfg["contrastive_weight"], cfg["consistency_weight"],
                )
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() / 2**30
            attempts.append({"microbatch": candidate, "status": "PASS", "peak_allocated_gib": peak})
            del model, optimizer, batch, target, negatives, first_prediction, second_prediction, loss
            torch.cuda.empty_cache()
            if peak <= cfg["peak_memory_limit_gib"]:
                return candidate, attempts
        except torch.cuda.OutOfMemoryError:
            attempts.append({"microbatch": candidate, "status": "OOM"})
            torch.cuda.empty_cache()
    raise RuntimeError(f"no posttraining microbatch passed: {attempts}")


def monitor_gpu(cfg, path: Path, stop: threading.Event):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("timestamp,gpu_util,memory_used_mib,memory_total_mib,temperature_c,power_w\n", encoding="utf-8")
    while not stop.is_set():
        result = subprocess.run([
            "nvidia-smi", f"--id={cfg['gpu_uuid']}",
            "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ], capture_output=True, text=True)
        if result.returncode == 0:
            with path.open("a", encoding="utf-8") as stream:
                stream.write(f"{time.time()},{result.stdout.strip()}\n")
        stop.wait(30)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        parser.error("--run is required")
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    if os.environ.get("CUDA_VISIBLE_DEVICES") != cfg["gpu_uuid"] or not torch.cuda.is_available():
        raise RuntimeError("D092 must run on the configured local GPU UUID")
    device = torch.device("cuda")
    data = common.load_subject_data(cfg)
    wave_mean, wave_scale, waveform = waveform_statistics(data["waveform"], data["fit"])
    frequency_mean, frequency_scale = frequency_statistics(waveform, data["fit"])
    targets, trial_targets, gallery_labels, target_contract, target_contract_path = load_targets(cfg, data)
    encoder, pretrain = encoder_from_best(cfg)
    encoder_state = encoder.state_dict()
    microbatch, attempts = capacity_probe(
        cfg, encoder_state, waveform, targets, trial_targets, frequency_mean.to(device), frequency_scale.to(device), device
    )
    report_dir = ROOT / cfg["report_dir"]
    run_dir = ROOT / cfg["run_dir"]
    report_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    split_evidence = common.split_evidence(data)
    contract = {
        "task": cfg["task"], "config_sha256": sha256(CONFIG),
        "model_sha256": sha256(ROOT / "src/d092_sample_efficient_latent.py"),
        "runner_sha256": sha256(Path(__file__)), "pretrain_sha256": sha256(ROOT / cfg["pretrain_best"]),
        "pretrain_step": int(pretrain["step"]), "pretrain_validation": pretrain["validation"],
        "target_contract_sha256": sha256(target_contract_path), "fit_hash": split_evidence["fit_hash"],
        "validation_hash": split_evidence["validation_hash"], "test_forward_count": 0,
    }
    preflight = {
        "status": "PASS", "contract": contract, "split": split_evidence,
        "targets": {"shape": list(targets.shape), "dtype": str(targets.dtype),
                    "latent_space": target_contract["latent_space"], "resident_in_system_ram": True},
        "statistics": {"wave_mean": tensor_sha256(wave_mean), "wave_scale": tensor_sha256(wave_scale),
                       "frequency_mean": tensor_sha256(frequency_mean), "frequency_scale": tensor_sha256(frequency_scale)},
        "capacity_attempts": attempts, "microbatch": microbatch, "logical_batch": cfg["logical_batch"],
        "loss": {"full_smooth_l1": 1.0, "token_cosine": cfg["cosine_weight"],
                 "train_only_contrastive": cfg["contrastive_weight"], "view_consistency": cfg["consistency_weight"]},
        "classification_head": False, "test_eeg_forward_count": 0,
    }
    atomic_json(report_dir / "preflight.json", preflight)

    model = SampleEfficientLatentPredictor(encoder, cfg["query_depth"], cfg["dropout"]).to(device)
    phase, stale, reductions, start_epoch = "head_only", 0, 0, 0
    set_phase(model, phase, cfg["phase_b_unfreeze_blocks"])
    optimizer = torch.optim.AdamW(optimizer_groups(model, cfg, phase))
    history, best_key = [], None
    latest_path, best_path = run_dir / "latest.pt", run_dir / "best.pt"
    if latest_path.exists():
        saved = torch.load(latest_path, map_location=device, weights_only=False)
        if saved["contract"] != contract:
            raise RuntimeError("D092 resume contract changed")
        phase, stale, reductions, start_epoch = saved["phase"], saved["stale"], saved["lr_reductions"], saved["epoch"]
        set_phase(model, phase, cfg["phase_b_unfreeze_blocks"])
        optimizer = torch.optim.AdamW(optimizer_groups(model, cfg, phase))
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        history, best_key = saved["history"], tuple(saved["best_key"])
        torch.set_rng_state(saved["torch_rng"].cpu())
        torch.cuda.set_rng_state_all([value.cpu() for value in saved["cuda_rng"]])
        np.random.set_state(saved["numpy_rng"])
    else:
        torch.manual_seed(cfg["seed"] + 9200)
        torch.cuda.manual_seed_all(cfg["seed"] + 9200)
        np.random.seed(cfg["seed"] + 9200)

    try:
        waveform = waveform.pin_memory()
        pinned = True
    except RuntimeError:
        pinned = False
    frequency_mean, frequency_scale = frequency_mean.to(device), frequency_scale.to(device)
    fit_targets = trial_targets[data["fit"]].numpy()
    validation_targets = trial_targets[data["validation"]]
    validation_labels = data["labels"][data["validation"]]
    validation_candidates = torch.unique(validation_targets, sorted=True)
    stop_monitor = threading.Event()
    monitor = threading.Thread(target=monitor_gpu, args=(cfg, report_dir / "gpu_samples.csv", stop_monitor), daemon=True)
    monitor.start()
    atomic_json(report_dir / "status.json", {
        "status": "training", "phase": phase, "epoch": start_epoch, "microbatch": microbatch,
        "waveform_pinned": pinned, "targets_in_system_ram": True, "test_forward_count": 0,
        "parameters": trainable_parameter_counts(model),
    })
    started = time.perf_counter()
    try:
        for epoch in range(start_epoch + 1, cfg["max_epochs_safety_cap"] + 1):
            if STOP:
                break
            model.train()
            if phase == "head_only":
                model.encoder.eval()
            order = data["fit"][torch.randperm(len(data["fit"]))]
            totals = {"loss": 0.0, "regression": 0.0, "token_cosine": 0.0, "contrastive": 0.0, "consistency": 0.0}
            seen = 0
            warm = min(1.0, epoch / 3.0)
            for group in optimizer.param_groups:
                base = cfg["encoder_lr"] if group.get("role") == "encoder." else cfg["head_peak_lr"]
                group["lr"] = base * warm * (0.5 ** reductions)
            for logical in order.split(cfg["logical_batch"]):
                optimizer.zero_grad(set_to_none=True)
                for ids in logical.split(microbatch):
                    batch = waveform[ids].to(device, non_blocking=True)
                    positive_indices = trial_targets[ids]
                    target = targets[positive_indices].to(device)
                    positive_set = set(int(value) for value in positive_indices)
                    available = np.asarray([value for value in fit_targets if int(value) not in positive_set])
                    negative_indices = np.random.choice(available, size=cfg["negative_targets"], replace=False)
                    negatives = targets[torch.from_numpy(negative_indices)].to(device)
                    first, first_mask = augment_view(batch, cfg)
                    second, second_mask = augment_view(batch, cfg)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        first_prediction = model(first, frequency_mean, frequency_scale, first_mask)
                        second_prediction = model(second, frequency_mean, frequency_scale, second_mask)
                        loss, parts = latent_objective(
                            first_prediction, second_prediction, target, negatives, cfg["temperature"],
                            cfg["cosine_weight"], cfg["contrastive_weight"], cfg["consistency_weight"],
                        )
                    weight = len(ids) / len(logical)
                    (loss * weight).backward()
                    totals["loss"] += float(loss.detach()) * len(ids)
                    for key, value in parts.items():
                        totals[key] += float(value) * len(ids)
                    seen += len(ids)
                grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"]))
                optimizer.step()

            validation_prediction = predict(
                model, waveform, data["validation"], frequency_mean, frequency_scale, microbatch, device
            )
            validation = retrieval_metrics(
                validation_prediction, validation_targets, validation_labels, validation_candidates,
                targets, gallery_labels, device,
            )
            key = selection_key(validation)
            improved = best_key is None or key > best_key
            stale = 0 if improved else stale + 1
            row = {
                "epoch": epoch, "phase": phase, "train": {key: value / seen for key, value in totals.items()},
                "validation_240_gallery": validation, "selection_key": list(key), "improved": improved,
                "stale": stale, "lr_reductions": reductions, "grad_norm": grad_norm,
                "learning_rates": [group["lr"] for group in optimizer.param_groups],
                "elapsed_seconds": time.perf_counter() - started,
                "test_forward_count": 0,
            }
            if epoch % cfg["full_gallery_every_epochs"] == 0:
                row["validation_4000_gallery"] = retrieval_metrics(
                    validation_prediction, validation_targets, validation_labels, torch.arange(4000),
                    targets, gallery_labels, device,
                )
            history.append(row)
            append_jsonl(run_dir / "curve.jsonl", row)
            if improved:
                best_key = key
                torch.save({"contract": contract, "epoch": epoch, "phase": phase, "model": model.state_dict(),
                            "validation": validation, "selection_key": list(key)}, best_path)

            if phase == "head_only" and epoch >= cfg["phase_a_min_epochs"] and stale >= cfg["phase_a_patience"]:
                phase, stale = "encoder_tail", 0
                set_phase(model, phase, cfg["phase_b_unfreeze_blocks"])
                optimizer = torch.optim.AdamW(optimizer_groups(model, cfg, phase))
                row["action"] = "unfreeze_encoder_tail"
            elif phase == "encoder_tail" and stale > 0 and stale % cfg["phase_b_lr_patience"] == 0 \
                    and reductions < cfg["phase_b_max_lr_reductions"]:
                reductions += 1
                row["action"] = "halve_learning_rates"

            state = {
                "contract": contract, "epoch": epoch, "phase": phase, "stale": stale,
                "lr_reductions": reductions, "best_key": list(best_key), "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "history": history,
                "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
                "numpy_rng": np.random.get_state(),
            }
            torch.save(state, latest_path.with_suffix(".tmp"))
            os.replace(latest_path.with_suffix(".tmp"), latest_path)
            if epoch % cfg["sparse_checkpoint_every_epochs"] == 0:
                torch.save(state, run_dir / f"epoch{epoch:03d}.pt")
            atomic_json(report_dir / "status.json", {
                "status": "training", "phase": phase, "epoch": epoch, "microbatch": microbatch,
                "latest_validation": validation, "best_key": list(best_key), "stale": stale,
                "lr_reductions": reductions, "test_forward_count": 0,
            })
            if phase == "encoder_tail" and stale >= cfg["phase_b_stop_patience"] \
                    and reductions >= cfg["phase_b_max_lr_reductions"]:
                break
        final = {
            "status": "complete" if not STOP else "paused", "epoch": history[-1]["epoch"] if history else start_epoch,
            "best": torch.load(best_path, map_location="cpu", weights_only=False)["validation"] if best_path.exists() else None,
            "best_checkpoint": str(best_path), "history_points": len(history), "test_forward_count": 0,
        }
        atomic_json(report_dir / "result.json", final)
        return 0
    finally:
        stop_monitor.set()
        monitor.join(timeout=2)


if __name__ == "__main__":
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, request_stop)
    try:
        raise SystemExit(main())
    except Exception as error:
        cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
        atomic_json(ROOT / cfg["report_dir"] / "status.json", {
            "status": "failed", "error_type": type(error).__name__, "error": str(error), "test_forward_count": 0,
        })
        raise
