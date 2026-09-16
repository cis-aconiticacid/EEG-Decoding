"""D079: coordinate-removal and learned early-channel-fusion A100 experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import signal
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
import d078_run_continuation as d078  # noqa: E402
import run_d064_mask_ratio as d064  # noqa: E402
import run_d066_latent_finetune as d066  # noqa: E402
from d079_architecture import (  # noqa: E402
    ARMS,
    EarlyFusionFrequency,
    NoCoordinateFrequency,
    architecture_record,
    load_compatible_parent,
    tensor_sha,
)
from eegdecoding.rae_latent_alignment import latent_loss  # noqa: E402


CFG_PATH = ROOT / "config/d079_no_coord_early_fusion.json"
STOP = False


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def emit(report: Path, phase: str, **fields: Any) -> None:
    record = {"task": "D079", "phase": phase, "pid": os.getpid(), "updated_unix": time.time(), **fields}
    atomic_json(report / "status.json", record)
    print(json.dumps(record, ensure_ascii=False, allow_nan=False), flush=True)


def guard(cfg: dict[str, Any]) -> dict[str, Any]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if sys.platform != "linux" or visible != cfg["gpu_uuid"]:
        raise RuntimeError("D079 GPU worker must run under the assigned UUID lock")
    lock = (ROOT / cfg["lock_root"] / f"gpu-{cfg['gpu_uuid']}.lock").resolve()
    if not lock.is_dir():
        raise RuntimeError("external UUID lock directory is absent")
    identified = False
    for _ in range(20):
        for metadata_path in lock.glob("*.json"):
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("worker_pid") == os.getpid() and metadata.get("queue_partition") == "gpu0":
                identified = True
                break
        if identified:
            break
        time.sleep(0.1)
    if not identified or torch.cuda.device_count() != 1 or "A100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("lock metadata or visible A100 identity differs")
    inventory = os.popen(
        "nvidia-smi -i 0 --query-gpu=index,uuid,name --format=csv,noheader,nounits"
    ).read().strip()
    if not inventory.startswith("0, " + cfg["gpu_uuid"]) or "A100" not in inventory:
        raise RuntimeError(f"physical GPU0 mapping differs: {inventory}")
    return {
        "visible": visible,
        "logical_device": 0,
        "name": torch.cuda.get_device_name(0),
        "physical_inventory": inventory,
    }


def build_model(arm: str, coordinates: torch.Tensor, parent_state: dict[str, torch.Tensor], seed: int):
    if arm == "B0":
        encoder = d064.MaskedFrequency(coordinates, seed)
    elif arm == "B1":
        encoder = NoCoordinateFrequency(seed)
    elif arm == "B2":
        encoder = EarlyFusionFrequency(seed)
    else:
        raise ValueError(f"unknown D079 arm: {arm}")
    model = d066.ImageDecoder(encoder, seed)
    adaptation = load_compatible_parent(model, parent_state, arm)
    return model, adaptation


def optimizer_for(model: nn.Module, arm: str, cfg: dict[str, Any]):
    fusion = [model.encoder.fusion] if arm == "B2" else []
    fusion_ids = {id(parameter) for parameter in fusion}
    encoder = [
        parameter for parameter in model.encoder.parameters()
        if parameter.requires_grad and id(parameter) not in fusion_ids
    ]
    head = [
        parameter for name, parameter in model.named_parameters()
        if parameter.requires_grad and not name.startswith("encoder.")
    ]
    groups: list[dict[str, Any]] = [
        {"params": encoder, "lr": cfg["encoder_lr"], "role": "existing_encoder"},
        {"params": head, "lr": cfg["head_lr"], "role": "image_head"},
    ]
    if fusion:
        groups.append({"params": fusion, "lr": cfg["fusion_lr"], "role": "fusion_P"})
    optimizer = torch.optim.AdamW(
        groups,
        weight_decay=cfg["weight_decay"],
        betas=tuple(cfg["betas"]),
        eps=cfg["eps"],
    )
    assigned = [parameter for group in groups for parameter in group["params"]]
    expected = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if len(assigned) != len(expected) or {id(parameter) for parameter in assigned} != {id(parameter) for parameter in expected}:
        raise AssertionError("optimizer parameter partition differs")
    return optimizer


def model_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def selected_hashes(model: nn.Module) -> dict[str, str]:
    result = {
        "projection": tensor_sha(model.encoder.projection.weight),
        "transformer": tensor_sha(model.encoder.layers[-1].linear2.weight),
        "image_head": tensor_sha(model.projector[-1].weight),
    }
    if hasattr(model.encoder, "fusion"):
        result["fusion_P"] = tensor_sha(model.encoder.fusion)
    return result


def checkpoint_payload(model, optimizer, arm, epoch, history, metrics, contract, architecture, cfg):
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "seed": cfg["seed"],
        "arm": arm,
        "contract": contract,
        "architecture": architecture,
        "history": history,
        "metrics": metrics,
        "cursor": {"completed_epoch": epoch, "next_epoch": epoch + 1, "completed_steps": (epoch - 70) * 38},
        "rng": {"torch_cpu": torch.get_rng_state(), "torch_cuda": torch.cuda.get_rng_state_all()},
    }


def evaluate_point(model, arm, epoch, x, train, test, ids, targets, gallery, labels, candidates,
                   records, rows, output, cfg):
    tick = time.perf_counter()
    test_metrics, predictions = d078.evaluate(
        model, x, test, ids, targets, gallery, labels, candidates, records, rows
    )
    clean = d078.clean_train_loss(model, x, train, ids, targets, cfg)
    prediction_path = output / "predictions" / f"epoch_{epoch:03d}.json"
    atomic_json(prediction_path, predictions)
    return {
        "arm": arm,
        "epoch": epoch,
        "clean_train_loss": clean["loss"],
        "clean_train_latent_mse": clean["latent_mse"],
        "clean_train_contrastive": clean["contrastive"],
        "test_class_accuracy": test_metrics["class_accuracy"],
        "test_image_top1": test_metrics["image_top1"],
        "test_latent_mse": test_metrics["latent_mse"],
        "test_n": test_metrics["n"],
        "prediction_sha256": sha(prediction_path),
        "model_state_sha256": model_hash(model),
        "evaluation_seconds": time.perf_counter() - tick,
    }


def write_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "arm", "epoch", "clean_train_loss", "clean_train_latent_mse", "clean_train_contrastive",
        "test_class_accuracy", "test_image_top1", "test_latent_mse", "test_n",
        "prediction_sha256", "model_state_sha256", "evaluation_seconds",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def endpoint_bootstrap(reference: list[dict[str, Any]], candidate: list[dict[str, Any]], seed: int):
    if [row["trial_id"] for row in reference] != [row["trial_id"] for row in candidate]:
        raise AssertionError("paired endpoint prediction order differs")
    groups: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(reference):
        groups[int(row["actual_label"])].append(index)
    keys = sorted(groups)
    if len(keys) != 80 or any(len(groups[key]) != 20 for key in keys):
        raise AssertionError("expected 80 classes with 20 test trials each")
    arrays = {
        "class_accuracy": np.asarray([row["class_correct"] for row in candidate]) - np.asarray([row["class_correct"] for row in reference]),
        "image_top1": np.asarray([row["image_correct"] for row in candidate]) - np.asarray([row["image_correct"] for row in reference]),
        "latent_mse": np.asarray([row["latent_mse"] for row in candidate]) - np.asarray([row["latent_mse"] for row in reference]),
    }
    generator = np.random.default_rng(seed)
    members = [np.asarray(groups[key]) for key in keys]
    samples = {key: np.empty(2000) for key in arrays}
    for replicate in range(2000):
        selected = generator.integers(0, 80, size=80)
        indices = np.concatenate([members[value] for value in selected])
        for key, values in arrays.items():
            samples[key][replicate] = values[indices].mean()
    return {
        key: {
            "observed_difference": float(values.mean()),
            "percentile_95ci": np.quantile(samples[key], [0.025, 0.975]).tolist(),
        }
        for key, values in arrays.items()
    }


def render_report(report: Path, rows: list[dict[str, Any]], architectures: list[dict[str, Any]], comparisons):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13.75, 10), constrained_layout=True)
    specifications = [
        ("test_class_accuracy", "Test class accuracy", 100.0, "Percent"),
        ("test_image_top1", "Test image Top-1", 100.0, "Percent"),
        ("test_latent_mse", "Test latent MSE", 1.0, "MSE"),
        ("clean_train_loss", "Clean train loss (eval, batch=64)", 1.0, "Loss"),
    ]
    colors = {"B0": "#333333", "B1": "#3274d9", "B2": "#e98a15"}
    for axis, (field, title, scale, ylabel) in zip(axes.flat, specifications):
        for arm in ARMS:
            arm_rows = [row for row in rows if row["arm"] == arm]
            axis.plot([row["epoch"] for row in arm_rows], [row[field] * scale for row in arm_rows],
                      marker="o", linewidth=2.2, label=arm, color=colors[arm])
        axis.set_title(title)
        axis.set_xlabel("Checkpoint epoch")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
    axes[0, 0].legend()
    fig.suptitle("D079 coordinate removal and early fusion (clean inputs)", fontsize=16)
    fig.savefig(report / "checkpoint_trends.png", dpi=180)
    plt.close(fig)

    architecture_lines = [
        "| Arm | Attention tokens | Total params | Trainable params | Coord params | Fusion params |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in architectures:
        architecture_lines.append(
            f"| {item['arm']} | {item['attention_tokens']} | {item['total_parameters']} | "
            f"{item['trainable_parameters']} | {item['coordinate_projection_parameters']} | {item['fusion_parameters']} |"
        )
    metric_lines = [
        "| Arm | Epoch | Clean train loss | Test class acc | Image Top-1 | Test latent MSE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        metric_lines.append(
            f"| {row['arm']} | {row['epoch']} | {row['clean_train_loss']:.6f} | "
            f"{100*row['test_class_accuracy']:.4f}% | {100*row['test_image_top1']:.4f}% | "
            f"{row['test_latent_mse']:.6f} |"
        )
    comparison_lines = []
    for name, result in comparisons.items():
        comparison_lines.append(
            f"- `{name}` class-accuracy difference: `{100*result['class_accuracy']['observed_difference']:.4f}` percentage points; "
            f"95% cluster bootstrap CI "
            f"`[{100*result['class_accuracy']['percentile_95ci'][0]:.4f}, "
            f"{100*result['class_accuracy']['percentile_95ci'][1]:.4f}]`."
        )
    text = "\n".join([
        "# D079 Coordinate and Early-Fusion Ablation", "",
        "Status: COMPLETE. Each arm starts from the D066 seed-17 epoch-70 model weights and receives a separate AdamW optimizer. No arm shares updates or optimizer state.", "",
        "Epoch 70 is the common parent endpoint; epochs 80, 90, 100, 110, and 120 are fixed continuation checkpoints. The historical 1,600-trial test set is used only for diagnostics, not for early stopping or checkpoint selection.", "",
        "B0 retains the parent architecture. B1 removes the learned `Linear(3,128)` coordinate bias. B2 adds a learned channel-by-patch table `P[16,62]` to B1 before frequency/coordinate fusion. The 16 rows are ordered temporal patches.", "",
        "## Architecture", "", *architecture_lines, "",
        "![D079 checkpoint trends](checkpoint_trends.png)", "",
        "## Checkpoint results", "", *metric_lines, "",
        "Clean training loss is measured in `model.eval()` with a fixed order and batch size 64. The objective is latent MSE plus 0.1 times symmetric contrastive loss.", "",
        "## Fixed epoch-120 comparisons", "", *comparison_lines, "",
        "B1 minus B0 estimates coordinate-removal sensitivity. B2 minus B1 estimates the contribution of the learned 16-by-62 table. Because all arms are warm-started, these results do not describe training from scratch.", "",
    ])
    (report / "RESULTS.md").write_text(text, encoding="utf-8")


def self_test() -> dict[str, Any]:
    torch.set_num_threads(4)
    coordinates = torch.randn(62, 3)
    parent = d066.ImageDecoder(d064.MaskedFrequency(coordinates, 17), 17).state_dict()
    records = []
    counts = {}
    for arm in ARMS:
        model, adaptation = build_model(arm, coordinates, parent, 17)
        model.eval()
        sample = torch.randn(1, 496, 4, requires_grad=arm == "B2")
        output = model(sample)
        if output.shape != (1, 256, 1024):
            raise AssertionError(f"D079 {arm} output shape differs")
        record = architecture_record(model, arm)
        record["adaptation"] = adaptation
        records.append(record)
        counts[arm] = record["total_parameters"]
        if arm == "B2":
            model.encoder.tokens(sample).square().mean().backward()
            channel_gradient = sample.grad.reshape(1, 62, 8, 4).abs().sum((0, 2, 3))
            if not bool((channel_gradient > 0).all()) or model.encoder.fusion.grad is None:
                raise AssertionError("B2 does not retain differentiable dependence on all 62 channels")
    if counts["B1"] != counts["B0"] - 512 or counts["B2"] != counts["B1"] + 992:
        raise AssertionError(f"D079 parameter deltas differ: {counts}")
    result = {"status": "PASS", "arms": records, "parameter_deltas": {"B1_minus_B0": -512, "B2_minus_B1": 992}}
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    test_result = self_test()
    if args.self_test or not args.train:
        return 0

    cfg = json.loads(CFG_PATH.read_text(encoding="utf-8"))
    if cfg["arms"] != list(ARMS) or cfg["evaluation_epochs"] != [70, 80, 90, 100, 110, 120]:
        raise AssertionError("D079 fixed arm/evaluation contract differs")
    if cfg["augmentation"] != "none" or (cfg["start_epoch"], cfg["end_epoch"], cfg["additional_epochs"]) != (70, 120, 50):
        raise AssertionError("D079 augmentation or epoch contract differs")
    report = ROOT / cfg["report_dir"]
    run_root = ROOT / cfg["run_dir"]
    report.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True, exist_ok=True)
    gpu = guard(cfg)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    parent = d078.load_parent(cfg)
    contract = {
        "config": sha(CFG_PATH),
        "runner": sha(Path(__file__)),
        "architecture": sha(ROOT / "src/d079_architecture.py"),
        "d078_data_eval": sha(ROOT / "scripts/d078_run_continuation.py"),
        "d064_runner": sha(ROOT / "scripts/run_d064_mask_ratio.py"),
        "d066_runner": sha(ROOT / "scripts/run_d066_latent_finetune.py"),
        "latent_module": sha(PROJECT_ROOT / "src/eegdecoding/rae_latent_alignment.py"),
        "manifest": sha(ROOT / cfg["manifest"]),
        "normalization": sha(ROOT / cfg["normalization"]),
        "parent": cfg["parent_checkpoint_sha256"],
    }
    contract_path = report / "contract.json"
    if contract_path.exists() and json.loads(contract_path.read_text(encoding="utf-8")) != contract:
        raise RuntimeError("D079 run contract changed")
    atomic_json(contract_path, contract)
    atomic_json(report / "self_test.json", test_result)

    emit(report, "loading_resident_data", gpu=gpu)
    x, train, test, coordinates, manifest_rows = d078.prepare_data(cfg, parent, report)
    targets, gallery, ids, labels, image_records, target_contract = d078.load_targets(cfg, manifest_rows)
    x, train, test, coordinates = (value.cuda() for value in (x, train, test, coordinates))
    candidates = ids[test]
    if len(candidates.unique()) != 1600:
        raise AssertionError("D079 fixed test gallery differs")
    resident = {
        "features_shape": list(x.shape),
        "targets_shape": list(targets.shape),
        "gallery_shape": list(gallery.shape),
        "features_device": str(x.device),
        "targets_device": str(targets.device),
        "gallery_device": str(gallery.device),
        "allocated_gib": torch.cuda.memory_allocated() / 2**30,
        "reserved_gib": torch.cuda.memory_reserved() / 2**30,
        "device": torch.cuda.get_device_name(0),
        "gpu_uuid": cfg["gpu_uuid"],
        "target_contract_sha256": sha(ROOT / cfg["target_dir"] / "targets.json"),
    }
    atomic_json(report / "resident_data.json", resident)
    emit(report, "resident_data_ready", **resident)

    all_metrics: list[dict[str, Any]] = []
    architectures: list[dict[str, Any]] = []
    arm_results: list[dict[str, Any]] = []
    for arm in ARMS:
        if STOP or (ROOT / ".runtime/d079_stop_requested").exists():
            return 76
        output = run_root / arm
        output.mkdir(parents=True, exist_ok=True)
        if (output / "result.json").exists():
            result = json.loads((output / "result.json").read_text(encoding="utf-8"))
            if result["contract"] != contract:
                raise RuntimeError(f"stale completed D079 arm: {arm}")
            arm_results.append(result)
            arm_metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
            all_metrics.extend(arm_metrics)
            architectures.append(result["architecture"])
            continue

        model, adaptation = build_model(arm, coordinates.cpu(), parent["model"], cfg["seed"])
        model = model.cuda()
        architecture = architecture_record(model, arm)
        architecture["parent_adaptation"] = adaptation
        initial_hashes = selected_hashes(model)
        initialization = {
            "arm": arm,
            "parent_checkpoint_sha256": cfg["parent_checkpoint_sha256"],
            "model_state_sha256": model_hash(model),
            "selected_parameter_hashes": initial_hashes,
            "architecture": architecture,
            "optimizer": "new AdamW with empty state",
        }
        initialization_path = output / "initialization.json"
        if initialization_path.exists():
            stored_initialization = json.loads(initialization_path.read_text(encoding="utf-8"))
            if stored_initialization != initialization:
                raise RuntimeError(f"D079 {arm} initialization changed")
        else:
            atomic_json(initialization_path, initialization)
        optimizer = optimizer_for(model, arm, cfg)
        if optimizer.state:
            raise AssertionError("new D079 optimizer unexpectedly has inherited state")

        history: list[dict[str, Any]] = []
        arm_metrics: list[dict[str, Any]] = []
        metrics_path = output / "metrics.json"
        if metrics_path.exists():
            arm_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        start_epoch = 70
        checkpoints = sorted(output.glob("epoch_*.pt"))
        if checkpoints:
            saved = torch.load(checkpoints[-1], map_location="cuda", weights_only=False)
            if saved["contract"] != contract or saved["arm"] != arm or saved["seed"] != cfg["seed"]:
                raise RuntimeError(f"D079 {arm} resume checkpoint differs")
            model.load_state_dict(saved["model"], strict=True)
            optimizer.load_state_dict(saved["optimizer"])
            history = saved["history"]
            start_epoch = saved["epoch"]
            del saved

        metric_epochs = {int(row["epoch"]) for row in arm_metrics}
        if start_epoch in cfg["evaluation_epochs"] and start_epoch not in metric_epochs:
            metric = evaluate_point(model, arm, start_epoch, x, train, test, ids, targets, gallery, labels,
                                    candidates, image_records, manifest_rows, output, cfg)
            arm_metrics.append(metric)
            atomic_json(metrics_path, arm_metrics)
            emit(report, "checkpoint_evaluated", **metric)
            if arm == "B0" and start_epoch == 70:
                expected = {"test_class_accuracy": 0.37125, "test_image_top1": 0.018125, "test_latent_mse": 0.9566891086101532}
                for key, value in expected.items():
                    if abs(metric[key] - value) > 5e-7:
                        raise AssertionError(f"B0 parent reproduction differs for {key}: {metric[key]} vs {value}")

        first_update: dict[str, Any] | None = None
        for epoch in range(start_epoch + 1, cfg["end_epoch"] + 1):
            if STOP or (ROOT / ".runtime/d079_stop_requested").exists():
                return 76
            tick = time.perf_counter()
            torch.manual_seed(cfg["seed"] * 100000 + epoch)
            torch.cuda.manual_seed_all(cfg["seed"] * 100000 + epoch)
            permutation = d064.order(cfg["seed"], epoch, cfg["train_count"]).cuda()
            model.train()
            total = mse_total = contrastive_total = 0.0
            seen = steps = 0
            for batch_indices in train[permutation].split(cfg["batch_size"]):
                before_first = selected_hashes(model) if first_update is None else None
                optimizer.zero_grad(set_to_none=True)
                prediction = model(x[batch_indices])
                loss, parts = latent_loss(
                    prediction,
                    targets[ids[batch_indices]],
                    ids[batch_indices],
                    temperature=cfg["temperature"],
                    contrastive_weight=cfg["contrastive_weight"],
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite D079 training loss")
                loss.backward()
                if first_update is None:
                    required = {
                        "projection": model.encoder.projection.weight.grad,
                        "transformer": model.encoder.layers[-1].linear2.weight.grad,
                        "image_head": model.projector[-1].weight.grad,
                    }
                    if arm == "B2":
                        required["fusion_P"] = model.encoder.fusion.grad
                    if any(value is None or not torch.isfinite(value).all() or float(value.abs().sum()) == 0.0 for value in required.values()):
                        raise AssertionError(f"D079 {arm} first-step gradient evidence differs")
                    gradient_norms = {name: float(value.norm()) for name, value in required.items()}
                nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"], error_if_nonfinite=True)
                optimizer.step()
                if first_update is None:
                    after_first = selected_hashes(model)
                    if any(before_first[name] == after_first[name] for name in required):
                        raise AssertionError(f"D079 {arm} required parameter did not update on first step")
                    first_update = {
                        "arm": arm,
                        "epoch": epoch,
                        "step": 1,
                        "loss": float(loss.detach()),
                        "gradient_norms_before_clip": gradient_norms,
                        "before_sha256": before_first,
                        "after_sha256": after_first,
                        "gpu_allocated_gib": torch.cuda.memory_allocated() / 2**30,
                        "gpu_reserved_gib": torch.cuda.memory_reserved() / 2**30,
                    }
                    atomic_json(output / "first_update.json", first_update)
                    emit(report, "first_update", **first_update)
                size = len(batch_indices)
                total += float(loss.detach()) * size
                mse_total += float(parts["latent_mse"]) * size
                contrastive_total += float(parts["contrastive"]) * size
                seen += size
                steps += 1
            torch.cuda.synchronize()
            if seen != cfg["train_count"] or steps != cfg["steps_per_epoch"]:
                raise AssertionError("D079 epoch coverage differs")
            epoch_record = {
                "epoch": epoch,
                "loss": total / seen,
                "latent_mse": mse_total / seen,
                "contrastive": contrastive_total / seen,
                "coverage": seen,
                "steps": steps,
                "sampler_sha256": tensor_sha(permutation),
                "dropout_seed": cfg["seed"] * 100000 + epoch,
                "encoder_lr": cfg["encoder_lr"],
                "head_lr": cfg["head_lr"],
                "fusion_lr": cfg["fusion_lr"] if arm == "B2" else None,
                "seconds": time.perf_counter() - tick,
            }
            history.append(epoch_record)
            atomic_json(output / "epochs.json", history)
            emit(report, "training", arm=arm, **epoch_record)
            if epoch in cfg["checkpoint_epochs"]:
                checkpoint_path = output / f"epoch_{epoch:03d}.pt"
                atomic_save(checkpoint_path, checkpoint_payload(
                    model, optimizer, arm, epoch, history, arm_metrics, contract, architecture, cfg
                ))
                metric = evaluate_point(model, arm, epoch, x, train, test, ids, targets, gallery, labels,
                                        candidates, image_records, manifest_rows, output, cfg)
                arm_metrics = [row for row in arm_metrics if int(row["epoch"]) != epoch] + [metric]
                arm_metrics.sort(key=lambda row: int(row["epoch"]))
                atomic_json(metrics_path, arm_metrics)
                emit(report, "checkpoint_evaluated", **metric)

        if len(list(output.glob("epoch_*.pt"))) != 5 or [row["epoch"] for row in arm_metrics] != cfg["evaluation_epochs"]:
            raise AssertionError(f"D079 {arm} checkpoint/evaluation count differs")
        final_hashes = selected_hashes(model)
        required_updates = ["projection", "transformer", "image_head"] + (["fusion_P"] if arm == "B2" else [])
        if any(final_hashes[name] == initial_hashes[name] for name in required_updates):
            raise AssertionError(f"D079 {arm} endpoint parameter update differs")
        result = {
            "arm": arm,
            "seed": cfg["seed"],
            "subject": cfg["subject"],
            "parent_epoch": 70,
            "epoch": 120,
            "additional_epochs": 50,
            "architecture": architecture,
            "optimizer": {
                "source": "new AdamW; no inherited state",
                "encoder_lr": cfg["encoder_lr"],
                "head_lr": cfg["head_lr"],
                "fusion_lr": cfg["fusion_lr"] if arm == "B2" else None,
                "weight_decay": cfg["weight_decay"],
                "betas": cfg["betas"],
                "eps": cfg["eps"],
            },
            "initial_parameter_sha256": initial_hashes,
            "final_parameter_sha256": final_hashes,
            "first_update": first_update,
            "metrics": arm_metrics,
            "checkpoint_count": 5,
            "contract": contract,
            "training_seconds": sum(row["seconds"] for row in history),
            "augmentation": "none",
        }
        atomic_json(output / "result.json", result)
        arm_results.append(result)
        all_metrics.extend(arm_metrics)
        architectures.append(architecture)
        atomic_json(report / "completed_arms.json", arm_results)
        del model, optimizer
        torch.cuda.empty_cache()

    all_metrics.sort(key=lambda row: (ARMS.index(row["arm"]), int(row["epoch"])))
    write_metrics_csv(report / "checkpoint_metrics.csv", all_metrics)
    comparisons = {}
    for reference_arm, candidate_arm in (("B0", "B1"), ("B1", "B2")):
        reference = json.loads((run_root / reference_arm / "predictions/epoch_120.json").read_text(encoding="utf-8"))
        candidate = json.loads((run_root / candidate_arm / "predictions/epoch_120.json").read_text(encoding="utf-8"))
        comparisons[f"{candidate_arm}_minus_{reference_arm}"] = endpoint_bootstrap(
            reference, candidate, cfg["seed"] * 100 + ARMS.index(candidate_arm)
        )
    summary = {
        "status": "complete",
        "contract": contract,
        "resident_data": resident,
        "architectures": architectures,
        "metrics": all_metrics,
        "comparisons_epoch120": comparisons,
        "test_policy": cfg["evaluation"]["test_policy"],
    }
    atomic_json(report / "checkpoint_metrics.json", summary)
    atomic_json(report / "comparisons_epoch120.json", comparisons)
    render_report(report, all_metrics, architectures, comparisons)
    atomic_json(run_root / "completed.json", {"status": "complete", "arms": list(ARMS), "contract": contract})
    emit(report, "completed", arms=len(ARMS), checkpoints=15, evaluations=18)
    return 0


def stop(*_: Any) -> None:
    global STOP
    STOP = True


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    raise SystemExit(main())
