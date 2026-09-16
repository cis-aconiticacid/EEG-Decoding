"""D080: train B0/B1/B2 from one shared random initialization on A100 GPU0."""

from __future__ import annotations

import argparse
import ast
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
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from eegdecoding.data import ensure_trial_manifest, load_part  # noqa: E402
from d079_architecture import architecture_record  # noqa: E402
from d080_from_scratch import (  # noqa: E402
    ARMS,
    ExactWarmupCosine,
    build_random_family,
    model_state_sha,
    scheduler_self_test,
    shared_initialization_evidence,
    tensor_sha,
)
from eegdecoding.rae_latent_alignment import latent_loss, normalized_flat  # noqa: E402


CFG_PATH = ROOT / "config/d080_from_scratch_no_coord.json"
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
    record = {"task": "D080", "phase": phase, "pid": os.getpid(), "updated_unix": time.time(), **fields}
    atomic_json(report / "status.json", record)
    print(json.dumps(record, ensure_ascii=False, allow_nan=False), flush=True)


def guard(cfg: dict[str, Any]) -> dict[str, Any]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if sys.platform != "linux" or visible != cfg["gpu_uuid"]:
        raise RuntimeError("D080 GPU worker must run under the assigned UUID lock")
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
        raise RuntimeError("D080 lock metadata or visible A100 identity differs")
    inventory = os.popen(
        "nvidia-smi -i 0 --query-gpu=index,uuid,name --format=csv,noheader,nounits"
    ).read().strip()
    if not inventory.startswith("0, " + cfg["gpu_uuid"]) or "A100" not in inventory:
        raise RuntimeError(f"physical GPU0 mapping differs: {inventory}")
    return {"visible": visible, "logical_device": 0, "name": torch.cuda.get_device_name(0), "physical_inventory": inventory}


def frequency(raw: torch.Tensor) -> torch.Tensor:
    window = torch.hann_window(400, periodic=True, dtype=raw.dtype, device=raw.device)
    transformed = torch.fft.rfft((raw - raw.mean(-1, keepdim=True)) * window)
    return (2 * transformed[..., 1:33].abs().square() / (1000 * window.square().sum())).clamp_min(1e-30).log10()


def sample_order(seed: int, epoch: int, count: int) -> torch.Tensor:
    return torch.randperm(count, generator=torch.Generator().manual_seed(seed * 100000 + epoch))


def prepare_data(cfg: dict[str, Any], report: Path):
    with (ROOT / cfg["manifest"]).open(newline="", encoding="utf-8") as stream:
        rows = sorted(
            [row for row in csv.DictReader(stream) if int(row["subject"]) == cfg["subject"]],
            key=lambda row: int(row["source_index"]),
        )
    if len(rows) != 4000:
        raise AssertionError("D080 subject0 count differs")
    for label in range(80):
        group = [row for row in rows if int(row["label_index"]) == label]
        if len(group) != 50 or len({row["image"] for row in group}) != 50:
            raise AssertionError(f"D080 class {label} trial/image count differs")
        if [int(row["position_in_class_block"]) for row in group] != list(range(50)):
            raise AssertionError(f"D080 class {label} order differs")
    train = torch.tensor([index for index, row in enumerate(rows) if int(row["position_in_class_block"]) < 30])
    test = torch.tensor([index for index, row in enumerate(rows) if int(row["position_in_class_block"]) >= 30])
    if (len(train), len(test)) != (2400, 1600):
        raise AssertionError("D080 train/test split count differs")
    train_images = {rows[index]["image"] for index in train.tolist()}
    test_images = {rows[index]["image"] for index in test.tolist()}
    if train_images & test_images:
        raise AssertionError("D080 train/test image overlap")

    archive = load_part(ROOT / cfg["archive"])
    raw = torch.empty(4000, 62, 400)
    for index, row in enumerate(rows):
        item = archive["dataset"][int(row["source_index"])]
        if int(item["subject"]) != 0 or str(item["image"]) != row["image"] or str(item["label"]) != row["label"]:
            raise AssertionError("D080 raw-to-manifest mapping differs")
        raw[index] = item["eeg_data"][:, 40:440].float()
    del archive
    if not torch.isfinite(raw).all():
        raise AssertionError("D080 raw input contains nonfinite values")
    spectral = frequency(raw)
    mean = spectral[train].mean(0, keepdim=True)
    std = spectral[train].std(0, correction=0, keepdim=True).clamp_min(1e-6)
    features = ((spectral - mean) / std).reshape(4000, 496, 4)
    if not torch.isfinite(features).all():
        raise AssertionError("D080 normalized features contain nonfinite values")

    with (ROOT / cfg["channel_map"]).open(newline="", encoding="utf-8") as stream:
        channels = sorted(csv.DictReader(stream), key=lambda row: int(row["tensor_index"]))
    if [int(row["tensor_index"]) for row in channels] != list(range(62)):
        raise AssertionError("D080 montage order differs")
    coordinates = torch.tensor([[float(row[f"{axis}_m_template_fit"]) for axis in "xyz"] for row in channels])
    coordinates -= coordinates.mean(0)
    coordinates /= torch.pdist(coordinates).median()

    normalization_path = report / "normalization.npz"
    temporary = normalization_path.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, mean=mean.numpy(), std=std.numpy())
    os.replace(temporary, normalization_path)
    split = {
        "train_trials": [rows[index]["trial_id"] for index in train.tolist()],
        "test_trials": [rows[index]["trial_id"] for index in test.tolist()],
        "train_images": sorted(train_images),
        "test_images": sorted(test_images),
        "image_intersection": [],
    }
    atomic_json(report / "split.json", split)
    data_contract = {
        "manifest_sha256": sha(ROOT / cfg["manifest"]),
        "montage_sha256": sha(ROOT / cfg["channel_map"]),
        "archive_name": cfg["archive"],
        "archive_size": (ROOT / cfg["archive"]).stat().st_size,
        "raw4000_sha256": tensor_sha(raw),
        "train_indices_sha256": tensor_sha(train),
        "test_indices_sha256": tensor_sha(test),
        "normalization_fit_trials": 2400,
        "normalization_test_trials": 0,
        "normalization_mean_sha256": tensor_sha(mean),
        "normalization_std_sha256": tensor_sha(std),
        "normalization_file_sha256": sha(normalization_path),
        "features_sha256": tensor_sha(features),
        "coordinates_sha256": tensor_sha(coordinates),
        "crop": [40, 440],
        "frequency_bins": [1, 32],
        "split_sha256": sha(report / "split.json"),
    }
    atomic_json(report / "data_contract.json", data_contract)
    del raw, spectral
    return features, train, test, coordinates, rows, split, data_contract, mean, std


def load_targets(cfg: dict[str, Any], rows: list[dict[str, str]]):
    target_contract_path = ROOT / cfg["target_dir"] / "targets.json"
    target_contract = json.loads(target_contract_path.read_text(encoding="utf-8"))
    teacher = json.loads((ROOT / cfg["teacher_config"]).read_text(encoding="utf-8"))
    if target_contract["shape_per_image"] != [256, 1024] or target_contract["all_image_count"] != 4000:
        raise AssertionError("D080 image target shape/count differs")
    if target_contract["identity"]["config_sha256"] != sha(ROOT / cfg["teacher_config"]):
        raise AssertionError("D080 teacher config identity differs")
    if target_contract["teacher_sha256"] != teacher["teacher_sha256"] or target_contract["stats_sha256"] != teacher["stats_sha256"]:
        raise AssertionError("D080 teacher asset identity differs")
    targets = torch.empty(4000, 256, 1024, device="cuda")
    records: list[dict[str, Any]] = []
    array_hashes = {}
    for split_name in ("train", "monitor_val", "final_holdout"):
        metadata = target_contract["splits"][split_name]
        path = ROOT / cfg["target_dir"] / f"{split_name}.npy"
        if not metadata["complete"] or sha(path) != metadata["array_sha256"]:
            raise AssertionError(f"D080 image target split differs: {split_name}")
        array_hashes[split_name] = metadata["array_sha256"]
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if list(array.shape) != metadata["shape"] or array.dtype != np.float32:
            raise AssertionError(f"D080 image target array differs: {split_name}")
        offset = len(records)
        for start in range(0, len(array), 64):
            stop = min(start + 64, len(array))
            targets[offset + start:offset + stop].copy_(torch.from_numpy(np.array(array[start:stop])))
        records.extend(metadata["images"])
    lookup = {record["image_id"]: index for index, record in enumerate(records)}
    if len(lookup) != 4000:
        raise AssertionError("D080 image target IDs are not unique")
    ids = torch.tensor([lookup[row["image"]] for row in rows], device="cuda")
    labels = torch.tensor([int(record["label_index"]) for record in records], device="cuda")
    for index, row in enumerate(rows):
        if int(labels[ids[index]]) != int(row["label_index"]):
            raise AssertionError("D080 image target label mapping differs")
    gallery = torch.empty(4000, 256 * 1024, device="cuda")
    for start in range(0, 4000, 64):
        stop = min(start + 64, 4000)
        if not torch.isfinite(targets[start:stop]).all():
            raise AssertionError("D080 image target contains nonfinite values")
        gallery[start:stop] = normalized_flat(targets[start:stop])
    image_contract = {
        "target_contract_sha256": sha(target_contract_path),
        "teacher_config_sha256": sha(ROOT / cfg["teacher_config"]),
        "teacher_weight_sha256": target_contract["teacher_sha256"],
        "teacher_stats_sha256": target_contract["stats_sha256"],
        "target_array_sha256": array_hashes,
        "shape": [4000, 256, 1024],
    }
    return targets, gallery, ids, labels, records, image_contract


@torch.no_grad()
def evaluate(model, features, indices, ids, targets, gallery, labels, candidates, records, rows):
    model.eval()
    predictions = []
    total_mse = 0.0
    class_correct = image_correct = 0
    for batch_indices in indices.split(32):
        predicted = model(features[batch_indices])
        sample_mse = (predicted - targets[ids[batch_indices]]).square().flatten(1).mean(1)
        total_mse += float(sample_mse.sum())
        query = normalized_flat(predicted)
        best = torch.full((len(batch_indices),), -float("inf"), device="cuda")
        winner = torch.zeros(len(batch_indices), dtype=torch.long, device="cuda")
        for chunk in candidates.split(256):
            scores = query @ gallery[chunk].T
            values, local = scores.max(1)
            take = values > best
            winner = torch.where(take, chunk[local], winner)
            best = torch.maximum(best, values)
        actual = ids[batch_indices]
        category_match = labels[winner] == labels[actual]
        image_match = winner == actual
        class_correct += int(category_match.sum())
        image_correct += int(image_match.sum())
        for local_index, source_index in enumerate(batch_indices.cpu().tolist()):
            selected = int(winner[local_index])
            predictions.append({
                "trial_id": rows[source_index]["trial_id"],
                "image": rows[source_index]["image"],
                "actual_label": int(rows[source_index]["label_index"]),
                "predicted_image": records[selected]["image_id"],
                "predicted_label": int(labels[selected]),
                "similarity": float(best[local_index]),
                "latent_mse": float(sample_mse[local_index]),
                "class_correct": int(category_match[local_index]),
                "image_correct": int(image_match[local_index]),
            })
    return {
        "n": len(indices),
        "latent_mse": total_mse / len(indices),
        "class_accuracy": class_correct / len(indices),
        "image_top1": image_correct / len(indices),
        "gallery_size": len(candidates),
    }, predictions


@torch.no_grad()
def clean_train_loss(model, features, train, ids, targets, cfg):
    model.eval()
    total = mse = contrastive = 0.0
    seen = 0
    for batch_indices in train.split(cfg["batch_size"]):
        loss, parts = latent_loss(
            model(features[batch_indices]), targets[ids[batch_indices]], ids[batch_indices],
            temperature=cfg["temperature"], contrastive_weight=cfg["contrastive_weight"],
        )
        count = len(batch_indices)
        total += float(loss) * count
        mse += float(parts["latent_mse"]) * count
        contrastive += float(parts["contrastive"]) * count
        seen += count
    return {"n": seen, "loss": total / seen, "latent_mse": mse / seen, "contrastive": contrastive / seen}


def selected_hashes(model: nn.Module) -> dict[str, str]:
    result = {
        "projection": tensor_sha(model.encoder.projection.weight),
        "electrode_embedding": tensor_sha(model.encoder.electrode.weight),
        "frequency_embedding": tensor_sha(model.encoder.frequency_position.weight),
        "transformer": tensor_sha(model.encoder.layers[-1].linear2.weight),
        "queries": tensor_sha(model.queries),
        "image_head": tensor_sha(model.projector[-1].weight),
    }
    if hasattr(model.encoder, "position"):
        result["coordinate_projection"] = tensor_sha(model.encoder.position.weight)
    if hasattr(model.encoder, "fusion"):
        result["fusion_P"] = tensor_sha(model.encoder.fusion)
    return result


def make_optimizer_scheduler(model: nn.Module, cfg: dict[str, Any]):
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        [{"params": parameters, "lr": 0.0, "role": "all_trainable"}],
        weight_decay=cfg["weight_decay"], betas=tuple(cfg["betas"]), eps=cfg["eps"],
    )
    scheduler = ExactWarmupCosine(
        optimizer, cfg["peak_lr"], cfg["min_lr"], cfg["warmup_steps"], cfg["total_steps"]
    )
    if optimizer.state:
        raise AssertionError("D080 fresh optimizer state is not empty")
    assigned = optimizer.param_groups[0]["params"]
    if {id(parameter) for parameter in assigned} != {id(parameter) for parameter in parameters}:
        raise AssertionError("D080 optimizer does not cover every trainable parameter")
    return optimizer, scheduler


def evaluate_point(model, arm, epoch, features, train, test, ids, targets, gallery, labels,
                   candidates, records, rows, output, cfg):
    tick = time.perf_counter()
    test_metrics, predictions = evaluate(model, features, test, ids, targets, gallery, labels, candidates, records, rows)
    clean = clean_train_loss(model, features, train, ids, targets, cfg)
    prediction_path = output / "predictions" / f"epoch_{epoch:03d}.json"
    atomic_json(prediction_path, predictions)
    return {
        "arm": arm,
        "epoch": epoch,
        "global_step": epoch * cfg["steps_per_epoch"],
        "clean_train_loss": clean["loss"],
        "clean_train_latent_mse": clean["latent_mse"],
        "clean_train_contrastive": clean["contrastive"],
        "test_class_accuracy": test_metrics["class_accuracy"],
        "test_image_top1": test_metrics["image_top1"],
        "test_latent_mse": test_metrics["latent_mse"],
        "test_n": test_metrics["n"],
        "prediction_sha256": sha(prediction_path),
        "model_state_sha256": model_state_sha(model.state_dict()),
        "evaluation_seconds": time.perf_counter() - tick,
    }


def checkpoint_payload(model, optimizer, scheduler, arm, epoch, history, metrics, contract,
                       architecture, initialization, split, mean, std, cfg):
    return {
        "task": "D080",
        "initialization_source": "random_initialization",
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "global_step": scheduler.completed_steps,
        "seed": cfg["seed"],
        "arm": arm,
        "config": cfg,
        "contract": contract,
        "architecture": architecture,
        "initialization": initialization,
        "history": history,
        "metrics": metrics,
        "split": split,
        "normalization": {"mean": mean, "std": std, "fit_trials": 2400, "test_trials": 0},
        "rng": {"torch_cpu": torch.get_rng_state(), "torch_cuda": torch.cuda.get_rng_state_all()},
    }


def write_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "arm", "epoch", "global_step", "clean_train_loss", "clean_train_latent_mse",
        "clean_train_contrastive", "test_class_accuracy", "test_image_top1", "test_latent_mse",
        "test_n", "prediction_sha256", "model_state_sha256", "evaluation_seconds",
    ]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def endpoint_bootstrap(reference: list[dict[str, Any]], candidate: list[dict[str, Any]], seed: int):
    if [row["trial_id"] for row in reference] != [row["trial_id"] for row in candidate]:
        raise AssertionError("D080 paired prediction order differs")
    groups: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(reference):
        groups[int(row["actual_label"])].append(index)
    keys = sorted(groups)
    if len(keys) != 80 or any(len(groups[key]) != 20 for key in keys):
        raise AssertionError("D080 expected 80 classes x20 test trials")
    arrays = {
        "class_accuracy": np.asarray([row["class_correct"] for row in candidate]) - np.asarray([row["class_correct"] for row in reference]),
        "image_top1": np.asarray([row["image_correct"] for row in candidate]) - np.asarray([row["image_correct"] for row in reference]),
        "latent_mse": np.asarray([row["latent_mse"] for row in candidate]) - np.asarray([row["latent_mse"] for row in reference]),
    }
    generator = np.random.default_rng(seed)
    members = [np.asarray(groups[key]) for key in keys]
    samples = {key: np.empty(2000) for key in arrays}
    for replicate in range(2000):
        chosen = generator.integers(0, 80, size=80)
        indices = np.concatenate([members[value] for value in chosen])
        for key, values in arrays.items():
            samples[key][replicate] = values[indices].mean()
    return {
        key: {
            "observed_difference": float(values.mean()),
            "percentile_95ci": np.quantile(samples[key], [0.025, 0.975]).tolist(),
        }
        for key, values in arrays.items()
    }


def render_report(report: Path, rows: list[dict[str, Any]], architectures, comparisons, initialization, cfg):
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
        axis.set_xlabel("Epoch")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
    axes[0, 0].legend()
    fig.suptitle("D080 from-scratch coordinate removal and early fusion", fontsize=16)
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
        "| Arm | Epoch | Step | Clean train loss | Test class acc | Image Top-1 | Test latent MSE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        metric_lines.append(
            f"| {row['arm']} | {row['epoch']} | {row['global_step']} | {row['clean_train_loss']:.6f} | "
            f"{100*row['test_class_accuracy']:.4f}% | {100*row['test_image_top1']:.4f}% | {row['test_latent_mse']:.6f} |"
        )
    comparison_lines = []
    for name, result in comparisons.items():
        comparison_lines.append(
            f"- `{name}` class-accuracy difference: `{100*result['class_accuracy']['observed_difference']:.4f}` percentage points; "
            f"95% cluster bootstrap CI `[{100*result['class_accuracy']['percentile_95ci'][0]:.4f}, "
            f"{100*result['class_accuracy']['percentile_95ci'][1]:.4f}]`."
        )
    text = "\n".join([
        "# D080 Architecture Ablation from Random Initialization", "",
        "Status: COMPLETE. The EEG encoder, Transformer, image head, queries, and embeddings for B0/B1/B2 start from a shared seed-17 random initialization. No trained D064, D066, D078, or D079 model or optimizer state is loaded.", "",
        f"The common learnable-state digest is `{initialization['shared_learnable_combined_sha256']}`. Shared tensors begin identically; the B2 patch-by-channel table is the only architecture-specific extra parameter. No pretrained RAE decoder state is loaded into the trainable EEG model.", "",
        "All arms train for a fixed 70 epochs on 2,400 subject-0 training trials. The 1,600 test trials contribute zero optimizer updates. There is no early stopping, model selection, masked pretraining, or continuation.", "",
        f"AdamW optimizes every trainable parameter. The learning rate warms up for {cfg['warmup_steps']} steps to {cfg['peak_lr']}, then follows a cosine schedule through step {cfg['total_steps']} to {cfg['min_lr']}.", "",
        "Epoch 0 and every 10 epochs are predeclared historical test diagnostics; they do not stop training or select a checkpoint. Each architecture ends at the fixed epoch-70 endpoint.", "",
        "## Architecture", "", *architecture_lines, "",
        "![D080 checkpoint trends](checkpoint_trends.png)", "",
        "## Checkpoint results", "", *metric_lines, "",
        "## Fixed epoch-70 comparisons", "", *comparison_lines, "",
    ])
    (report / "RESULTS.md").write_text(text, encoding="utf-8")


def self_test() -> dict[str, Any]:
    torch.set_num_threads(4)
    coordinates = torch.randn(62, 3)
    models, copied = build_random_family(coordinates, 17)
    shared = shared_initialization_evidence(models)
    records = []
    for arm, model in models.items():
        model.eval()
        sample = torch.randn(1, 496, 4, requires_grad=arm == "B2")
        output = model(sample)
        if output.shape != (1, 256, 1024):
            raise AssertionError(f"D080 {arm} output shape differs")
        record = architecture_record(model, arm)
        state_names = set(model.state_dict())
        if arm in {"B1", "B2"} and any(
            "coordinates" in name or ("position" in name and "frequency_position" not in name)
            for name in state_names
        ):
            raise AssertionError(f"D080 {arm} retains coordinate state")
        if arm == "B2":
            model.encoder.tokens(sample).square().mean().backward()
            channel_gradient = sample.grad.reshape(1, 62, 8, 4).abs().sum((0, 2, 3))
            if not bool((channel_gradient > 0).all()) or model.encoder.fusion.grad is None:
                raise AssertionError("D080 B2 does not depend on all 62 input channels")
        records.append(record)
    optimizer, scheduler = make_optimizer_scheduler(models["B2"], {
        "peak_lr": 3e-4, "min_lr": 3e-5, "warmup_steps": 190, "total_steps": 2660,
        "weight_decay": 0.001, "betas": [0.9, 0.999], "eps": 1e-8,
    })
    if optimizer.state or scheduler.completed_steps != 0:
        raise AssertionError("D080 optimizer/scheduler did not start empty")
    schedule = scheduler_self_test()
    source = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    load_calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "torch"
        and node.func.attr == "load"
    ]
    if len(load_calls) != 1 or ast.unparse(load_calls[0].args[0]) != "checkpoint_path":
        raise AssertionError("D080 torch.load is not restricted to its own resume checkpoint")
    result = {
        "status": "PASS",
        "architectures": records,
        "shared_random_initialization": {
            key: value for key, value in shared.items() if key not in {"hashes", "state_names", "shared_learnable_names"}
        },
        "copied_state_entry_counts": {arm: len(names) for arm, names in copied.items()},
        "optimizer_initial_state_entries": 0,
        "scheduler": schedule,
        "forbidden_trained_parent_references": [],
    }
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
    ensure_trial_manifest()
    if cfg["arms"] != list(ARMS) or cfg["checkpoint_epochs"] != [10, 20, 30, 40, 50, 60, 70]:
        raise AssertionError("D080 arm/checkpoint contract differs")
    if cfg["augmentation"] != "none" or cfg["total_steps"] != 2660 or cfg["warmup_steps"] != 190:
        raise AssertionError("D080 augmentation/schedule contract differs")
    report = ROOT / cfg["report_dir"]
    run_root = ROOT / cfg["run_dir"]
    report.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True, exist_ok=True)
    gpu = guard(cfg)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    emit(report, "preparing_train_only_normalization", gpu=gpu)
    features, train, test, coordinates, rows, split, data_contract, mean, std = prepare_data(cfg, report)
    emit(report, "loading_resident_image_targets")
    targets, gallery, ids, labels, records, image_contract = load_targets(cfg, rows)
    features, train, test = (value.cuda() for value in (features, train, test))
    candidates = ids[test]
    if len(candidates.unique()) != 1600:
        raise AssertionError("D080 test gallery differs")
    resident = {
        "features_shape": list(features.shape), "targets_shape": list(targets.shape), "gallery_shape": list(gallery.shape),
        "features_device": str(features.device), "targets_device": str(targets.device), "gallery_device": str(gallery.device),
        "allocated_gib": torch.cuda.memory_allocated() / 2**30, "reserved_gib": torch.cuda.memory_reserved() / 2**30,
        "device": torch.cuda.get_device_name(0), "gpu_uuid": cfg["gpu_uuid"],
    }
    atomic_json(report / "resident_data.json", resident)
    emit(report, "resident_data_ready", **resident)

    models, copied = build_random_family(coordinates, cfg["seed"])
    shared_initialization = shared_initialization_evidence(models)
    initialization_summary = {
        "source": "random_initialization",
        "seed": cfg["seed"],
        "trained_checkpoint_load_count": 0,
        "optimizer_state_entries_before_training": {arm: 0 for arm in ARMS},
        "shared": shared_initialization,
        "copied_state_names": copied,
        "B2_fusion_initial_sha256": tensor_sha(models["B2"].encoder.fusion),
        "B2_fusion_row_orthogonality_max_error": float(
            (models["B2"].encoder.fusion @ models["B2"].encoder.fusion.T - torch.eye(16)).abs().max()
        ),
        "B1_B2_coordinate_state_absent": True,
        "external_supervision": "frozen RAE target arrays only",
    }
    atomic_json(report / "initialization.json", initialization_summary)
    atomic_json(report / "self_test.json", test_result)

    contract = {
        "config_sha256": sha(CFG_PATH),
        "runner_sha256": sha(Path(__file__)),
        "d080_module_sha256": sha(ROOT / "src/d080_from_scratch.py"),
        "d079_architecture_sha256": sha(ROOT / "src/d079_architecture.py"),
        "latent_module_sha256": sha(PROJECT_ROOT / "src/eegdecoding/rae_latent_alignment.py"),
        "loader_sha256": sha(PROJECT_ROOT / "src/eegdecoding/data.py"),
        "data": data_contract,
        "image_assets": image_contract,
        "initialization_source": "random_initialization",
        "shared_learnable_initial_sha256": shared_initialization["shared_learnable_combined_sha256"],
        "forbidden_trained_checkpoint_loads": [],
    }
    contract_path = report / "contract.json"
    if contract_path.exists() and json.loads(contract_path.read_text(encoding="utf-8")) != contract:
        raise RuntimeError("D080 run contract changed")
    atomic_json(contract_path, contract)

    all_metrics = []
    architectures = []
    results = []
    for arm in ARMS:
        if STOP or (ROOT / ".runtime/d080_stop_requested").exists():
            return 76
        output = run_root / arm
        output.mkdir(parents=True, exist_ok=True)
        if (output / "result.json").exists():
            result = json.loads((output / "result.json").read_text(encoding="utf-8"))
            if result["contract"] != contract:
                raise RuntimeError(f"stale completed D080 arm: {arm}")
            results.append(result)
            all_metrics.extend(result["metrics"])
            architectures.append(result["architecture"])
            del models[arm]
            continue

        model = models.pop(arm).cuda()
        architecture = architecture_record(model, arm)
        initial_hashes = selected_hashes(model)
        arm_initialization = {
            "source": "random_initialization",
            "seed": cfg["seed"],
            "model_state_sha256": model_state_sha(model.state_dict()),
            "selected_parameter_sha256": initial_hashes,
            "shared_learnable_initial_sha256": shared_initialization["shared_learnable_combined_sha256"],
            "trained_checkpoint_load_count": 0,
            "architecture": architecture,
        }
        atomic_json(output / "initialization.json", arm_initialization)
        optimizer, scheduler = make_optimizer_scheduler(model, cfg)
        history = []
        metrics = []
        start_epoch = 0
        checkpoints = sorted(output.glob("epoch_*.pt"))
        if checkpoints:
            checkpoint_path = checkpoints[-1]
            if run_root not in checkpoint_path.resolve().parents:
                raise RuntimeError("D080 attempted to resume outside its own run directory")
            saved = torch.load(checkpoint_path, map_location="cuda", weights_only=False)
            if saved["task"] != "D080" or saved["initialization_source"] != "random_initialization":
                raise RuntimeError("D080 resume checkpoint is not a D080 random initialization run")
            if saved["contract"] != contract or saved["arm"] != arm or saved["seed"] != cfg["seed"]:
                raise RuntimeError("D080 resume checkpoint contract differs")
            model.load_state_dict(saved["model"], strict=True)
            optimizer.load_state_dict(saved["optimizer"])
            scheduler.load_state_dict(saved["scheduler"])
            history = saved["history"]
            metrics = saved["metrics"]
            start_epoch = saved["epoch"]
            del saved
        elif optimizer.state or scheduler.completed_steps != 0:
            raise AssertionError("D080 fresh optimizer/scheduler state differs")

        if start_epoch == 0 and not metrics:
            metric = evaluate_point(model, arm, 0, features, train, test, ids, targets, gallery, labels,
                                    candidates, records, rows, output, cfg)
            metrics.append(metric)
            atomic_json(output / "metrics.json", metrics)
            emit(report, "checkpoint_evaluated", **metric)

        first_update_path = output / "first_update.json"
        first_update = json.loads(first_update_path.read_text(encoding="utf-8")) if first_update_path.exists() else None
        for epoch in range(start_epoch + 1, cfg["epochs"] + 1):
            if STOP or (ROOT / ".runtime/d080_stop_requested").exists():
                return 76
            tick = time.perf_counter()
            dropout_seed = cfg["seed"] * 100000 + epoch
            torch.manual_seed(dropout_seed)
            torch.cuda.manual_seed_all(dropout_seed)
            permutation = sample_order(cfg["seed"], epoch, cfg["train_count"]).cuda()
            model.train()
            total = mse_total = contrastive_total = 0.0
            seen = steps = 0
            first_lr = last_lr = None
            for batch_indices in train[permutation].split(cfg["batch_size"]):
                learning_rate = scheduler.step_before_optimizer()
                if first_lr is None:
                    first_lr = learning_rate
                last_lr = learning_rate
                before_first = selected_hashes(model) if first_update is None else None
                optimizer.zero_grad(set_to_none=True)
                loss, parts = latent_loss(
                    model(features[batch_indices]), targets[ids[batch_indices]], ids[batch_indices],
                    temperature=cfg["temperature"], contrastive_weight=cfg["contrastive_weight"],
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite D080 training loss")
                loss.backward()
                if first_update is None:
                    required = {
                        "projection": model.encoder.projection.weight.grad,
                        "electrode_embedding": model.encoder.electrode.weight.grad,
                        "frequency_embedding": model.encoder.frequency_position.weight.grad,
                        "transformer": model.encoder.layers[-1].linear2.weight.grad,
                        "queries": model.queries.grad,
                        "image_head": model.projector[-1].weight.grad,
                    }
                    if arm == "B0":
                        required["coordinate_projection"] = model.encoder.position.weight.grad
                    if arm == "B2":
                        required["fusion_P"] = model.encoder.fusion.grad
                    if any(value is None or not torch.isfinite(value).all() or float(value.abs().sum()) == 0.0 for value in required.values()):
                        raise AssertionError(f"D080 {arm} first-step gradient evidence differs")
                    gradient_norms = {name: float(value.norm()) for name, value in required.items()}
                nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"], error_if_nonfinite=True)
                optimizer.step()
                if first_update is None:
                    after_first = selected_hashes(model)
                    if any(before_first[name] == after_first[name] for name in required):
                        raise AssertionError(f"D080 {arm} required parameter did not update on first step")
                    if scheduler.completed_steps != 1 or learning_rate != cfg["peak_lr"] / cfg["warmup_steps"]:
                        raise AssertionError("D080 first-step LR differs")
                    first_update = {
                        "arm": arm, "epoch": 1, "global_step": 1, "loss": float(loss.detach()), "lr": learning_rate,
                        "gradient_norms_before_clip": gradient_norms, "before_sha256": before_first,
                        "after_sha256": after_first, "gpu_allocated_gib": torch.cuda.memory_allocated() / 2**30,
                        "gpu_reserved_gib": torch.cuda.memory_reserved() / 2**30,
                    }
                    atomic_json(first_update_path, first_update)
                    emit(report, "first_update", **first_update)
                count = len(batch_indices)
                total += float(loss.detach()) * count
                mse_total += float(parts["latent_mse"]) * count
                contrastive_total += float(parts["contrastive"]) * count
                seen += count
                steps += 1
            torch.cuda.synchronize()
            if seen != 2400 or steps != 38 or scheduler.completed_steps != epoch * 38:
                raise AssertionError("D080 epoch coverage/global step differs")
            epoch_record = {
                "epoch": epoch, "global_step_start": (epoch - 1) * 38 + 1, "global_step_end": epoch * 38,
                "loss": total / seen, "latent_mse": mse_total / seen, "contrastive": contrastive_total / seen,
                "coverage": seen, "steps": steps, "sampler_sha256": tensor_sha(permutation),
                "dropout_seed": dropout_seed, "lr_first": first_lr, "lr_last": last_lr,
                "seconds": time.perf_counter() - tick,
            }
            history.append(epoch_record)
            atomic_json(output / "epochs.json", history)
            emit(report, "training", arm=arm, **epoch_record)
            if epoch in cfg["checkpoint_epochs"]:
                metric = evaluate_point(model, arm, epoch, features, train, test, ids, targets, gallery, labels,
                                        candidates, records, rows, output, cfg)
                metrics.append(metric)
                atomic_json(output / "metrics.json", metrics)
                checkpoint = checkpoint_payload(
                    model, optimizer, scheduler, arm, epoch, history, metrics, contract, architecture,
                    arm_initialization, split, mean, std, cfg,
                )
                atomic_save(output / f"epoch_{epoch:03d}.pt", checkpoint)
                emit(report, "checkpoint_evaluated", **metric)

        if scheduler.completed_steps != cfg["total_steps"] or abs(scheduler.lr_at(cfg["total_steps"]) - cfg["min_lr"]) > 1e-18:
            raise AssertionError("D080 final scheduler endpoint differs")
        if len(list(output.glob("epoch_*.pt"))) != 7 or [row["epoch"] for row in metrics] != cfg["evaluation_epochs"]:
            raise AssertionError(f"D080 {arm} checkpoint/evaluation count differs")
        final_hashes = selected_hashes(model)
        required_updates = set(initial_hashes)
        if any(final_hashes[name] == initial_hashes[name] for name in required_updates):
            raise AssertionError(f"D080 {arm} endpoint parameter update differs")
        result = {
            "arm": arm, "seed": cfg["seed"], "subject": 0, "initialization_source": "random_initialization",
            "epochs": 70, "global_steps": scheduler.completed_steps, "architecture": architecture,
            "optimizer": {"source": "new AdamW", "parameter_groups": 1, "peak_lr": cfg["peak_lr"],
                          "min_lr": cfg["min_lr"], "warmup_steps": cfg["warmup_steps"],
                          "weight_decay": cfg["weight_decay"], "betas": cfg["betas"], "eps": cfg["eps"]},
            "scheduler": scheduler.state_dict(), "initial_parameter_sha256": initial_hashes,
            "final_parameter_sha256": final_hashes, "first_update": first_update, "metrics": metrics,
            "checkpoint_count": 7, "contract": contract, "training_seconds": sum(row["seconds"] for row in history),
            "augmentation": "none", "normalization_fit_trials": 2400, "normalization_test_trials": 0,
            "frozen_parameters": ["encoder.mask_token", "encoder.output.weight", "encoder.output.bias"],
        }
        atomic_json(output / "result.json", result)
        results.append(result)
        all_metrics.extend(metrics)
        architectures.append(architecture)
        atomic_json(report / "completed_arms.json", results)
        del model, optimizer, scheduler
        torch.cuda.empty_cache()

    all_metrics.sort(key=lambda row: (ARMS.index(row["arm"]), row["epoch"]))
    write_metrics_csv(report / "checkpoint_metrics.csv", all_metrics)
    comparisons = {}
    for reference_arm, candidate_arm in (("B0", "B1"), ("B1", "B2")):
        reference = json.loads((run_root / reference_arm / "predictions/epoch_070.json").read_text(encoding="utf-8"))
        candidate = json.loads((run_root / candidate_arm / "predictions/epoch_070.json").read_text(encoding="utf-8"))
        comparisons[f"{candidate_arm}_minus_{reference_arm}"] = endpoint_bootstrap(
            reference, candidate, cfg["seed"] * 100 + ARMS.index(candidate_arm)
        )
    summary = {
        "status": "complete", "contract": contract, "resident_data": resident,
        "initialization": initialization_summary, "architectures": architectures,
        "metrics": all_metrics, "comparisons_epoch70": comparisons, "test_policy": cfg["test_policy"],
    }
    atomic_json(report / "checkpoint_metrics.json", summary)
    atomic_json(report / "comparisons_epoch70.json", comparisons)
    render_report(report, all_metrics, architectures, comparisons, shared_initialization, cfg)
    atomic_json(run_root / "completed.json", {"status": "complete", "arms": list(ARMS), "contract": contract})
    emit(report, "completed", arms=3, checkpoints=21, evaluations=24, global_steps_per_arm=2660)
    return 0


def stop(*_: Any) -> None:
    global STOP
    STOP = True


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    raise SystemExit(main())
