"""D078: continue D066 seed17 on four fixed augmentation arms through epoch120."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import signal
import statistics
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
import run_d064_mask_ratio as d064  # noqa: E402
import run_d066_latent_finetune as d066  # noqa: E402
from d078_augmentation import ARMS, apply_augmentation, self_test as augmentation_self_test  # noqa: E402
from eegdecoding.data import ensure_trial_manifest, load_part  # noqa: E402
from eegdecoding.rae_latent_alignment import latent_loss, normalized_flat  # noqa: E402

CFG_PATH = ROOT / "config/d078_a100_continuation.json"
STOP = False


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def thash(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def atomic_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temp)
    os.replace(temp, path)


def emit(report: Path, phase: str, **fields: Any) -> None:
    value = {"task": "D078", "phase": phase, "pid": os.getpid(), "updated_unix": time.time(), **fields}
    atomic_json(report / "status.json", value)
    print(json.dumps(value, ensure_ascii=False, allow_nan=False), flush=True)


def guard(cfg: dict[str, Any]) -> dict[str, Any]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if sys.platform != "linux" or visible != cfg["gpu_uuid"]:
        raise RuntimeError("D078 GPU worker must run under the assigned UUID lock")
    lock = (ROOT / cfg["lock_root"] / f"gpu-{cfg['gpu_uuid']}.lock").resolve()
    if not lock.is_dir():
        raise RuntimeError("external UUID lock directory is absent")
    identified = False
    for _ in range(20):
        for meta_path in lock.glob("*.json"):
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("worker_pid") == os.getpid() and meta.get("queue_partition") == "gpu0":
                identified = True
                break
        if identified:
            break
        time.sleep(0.1)
    if not identified or torch.cuda.device_count() != 1 or "A100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("lock metadata or visible A100 identity differs")
    row = os.popen("nvidia-smi -i 0 --query-gpu=index,uuid,name --format=csv,noheader,nounits").read().strip()
    if not row.startswith("0, " + cfg["gpu_uuid"]) or "A100" not in row:
        raise RuntimeError(f"physical GPU0 mapping differs: {row}")
    return {"visible": visible, "logical_device": 0, "name": torch.cuda.get_device_name(0), "physical_inventory": row}


def load_parent(cfg: dict[str, Any]) -> dict[str, Any]:
    path = ROOT / cfg["parent_checkpoint"]
    if sha(path) != cfg["parent_checkpoint_sha256"]:
        raise RuntimeError("D066 parent checkpoint SHA256 changed")
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved.get("epoch") != 70 or saved.get("seed") != 17 or set(saved) != {"model", "optimizer", "epoch", "seed", "contract", "history"}:
        raise RuntimeError("D066 parent checkpoint contract differs")
    groups = saved["optimizer"]["param_groups"]
    if len(groups) != 2 or groups[0]["lr"] != 3e-6 or groups[1]["lr"] != 3e-5 or any(group["weight_decay"] != 0.001 for group in groups):
        raise RuntimeError("D066 parent optimizer endpoint differs")
    if sum(value.numel() for value in saved["model"].values()) != 1462462:
        raise RuntimeError("D066 parent model state size differs")
    return saved


def prepare_data(cfg: dict[str, Any], parent: dict[str, Any], report: Path):
    d064_cfg = json.loads((ROOT / cfg["d064_config"]).read_text(encoding="utf-8"))
    with (ROOT / cfg["manifest"]).open(newline="", encoding="utf-8") as stream:
        rows = sorted([row for row in csv.DictReader(stream) if int(row["subject"]) == 0], key=lambda row: int(row["source_index"]))
    if len(rows) != 4000:
        raise AssertionError("subject0 trial count differs")
    train = torch.tensor([index for index, row in enumerate(rows) if int(row["position_in_class_block"]) < 30])
    test = torch.tensor([index for index, row in enumerate(rows) if int(row["position_in_class_block"]) >= 30])
    if (len(train), len(test)) != (2400, 1600):
        raise AssertionError("subject0 train/test count differs")
    train_images = {rows[index]["image"] for index in train.tolist()}
    test_images = {rows[index]["image"] for index in test.tolist()}
    if train_images & test_images:
        raise AssertionError("subject0 image split overlaps")

    archive = load_part(ROOT / d064_cfg["archive"])
    raw = torch.empty(4000, 62, 400)
    for index, row in enumerate(rows):
        item = archive["dataset"][int(row["source_index"])]
        if int(item["subject"]) != 0 or str(item["image"]) != row["image"] or str(item["label"]) != row["label"]:
            raise AssertionError("subject0 raw-to-manifest mapping differs")
        raw[index] = item["eeg_data"][:, 40:440].float()
    del archive
    if thash(raw) != parent["contract"]["raw4000"] or thash(train) != parent["contract"]["train"] or thash(test) != parent["contract"]["test"]:
        raise AssertionError("D066 parent raw/split provenance differs")
    stats = torch.load(ROOT / cfg["normalization"], map_location="cpu", weights_only=True)
    x = ((d064.frequency(raw) - stats["mean"]) / stats["std"]).reshape(4000, 496, 4)
    del raw

    with (ROOT / d064_cfg["channel_map"]).open(newline="", encoding="utf-8") as stream:
        channels = sorted(csv.DictReader(stream), key=lambda row: int(row["tensor_index"]))
    coords = torch.tensor([[float(row[f"{axis}_m_template_fit"]) for axis in "xyz"] for row in channels])
    coords -= coords.mean(0)
    coords /= torch.pdist(coords).median()
    if not torch.isfinite(x).all() or not torch.isfinite(coords).all():
        raise AssertionError("nonfinite D078 input or coordinates")
    atomic_json(report / "split.json", {
        "train_trials": [rows[index]["trial_id"] for index in train.tolist()],
        "test_trials": [rows[index]["trial_id"] for index in test.tolist()],
        "train_images": sorted(train_images), "test_images": sorted(test_images), "image_intersection": [],
    })
    return x, train, test, coords, rows


def load_targets(cfg: dict[str, Any], rows: list[dict[str, str]]):
    contract_path = ROOT / cfg["target_dir"] / "targets.json"
    target_contract = json.loads(contract_path.read_text(encoding="utf-8"))
    teacher = json.loads((ROOT / cfg["teacher_config"]).read_text(encoding="utf-8"))
    if target_contract["shape_per_image"] != [256, 1024] or target_contract["all_image_count"] != 4000:
        raise AssertionError("image target shape/count differs")
    if target_contract["identity"]["config_sha256"] != sha(ROOT / cfg["teacher_config"]):
        raise AssertionError("teacher config identity differs")
    if target_contract["teacher_sha256"] != teacher["teacher_sha256"] or target_contract["stats_sha256"] != teacher["stats_sha256"]:
        raise AssertionError("teacher target hashes differ")
    targets = torch.empty(4000, 256, 1024, device="cuda")
    records: list[dict[str, Any]] = []
    for split in ("train", "monitor_val", "final_holdout"):
        meta = target_contract["splits"][split]
        path = ROOT / cfg["target_dir"] / f"{split}.npy"
        if not meta["complete"] or sha(path) != meta["array_sha256"]:
            raise AssertionError(f"target split differs: {split}")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if list(array.shape) != meta["shape"] or array.dtype != np.float32:
            raise AssertionError(f"target array differs: {split}")
        offset = len(records)
        for start in range(0, len(array), 64):
            stop = min(start + 64, len(array))
            targets[offset + start : offset + stop].copy_(torch.from_numpy(np.array(array[start:stop])))
        records.extend(meta["images"])
    lookup = {record["image_id"]: index for index, record in enumerate(records)}
    if len(lookup) != 4000:
        raise AssertionError("target image IDs are not unique")
    ids = torch.tensor([lookup[row["image"]] for row in rows], device="cuda")
    labels = torch.tensor([int(record["label_index"]) for record in records], device="cuda")
    gallery = torch.empty(4000, 256 * 1024, device="cuda")
    for start in range(0, 4000, 64):
        stop = min(start + 64, 4000)
        if not torch.isfinite(targets[start:stop]).all():
            raise AssertionError("nonfinite target")
        gallery[start:stop] = normalized_flat(targets[start:stop])
    return targets, gallery, ids, labels, records, target_contract


def build_model_optimizer(coords: torch.Tensor, parent: dict[str, Any], cfg: dict[str, Any]):
    model = d066.ImageDecoder(d064.MaskedFrequency(coords, int(cfg["seed"])), int(cfg["seed"]))
    model.load_state_dict(parent["model"], strict=True)
    model = model.cuda()
    encoder_parameters = [parameter for parameter in model.encoder.parameters() if parameter.requires_grad]
    head_parameters = [parameter for name, parameter in model.named_parameters() if not name.startswith("encoder.") and parameter.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": encoder_parameters, "lr": cfg["encoder_lr"], "scale": 0.1},
        {"params": head_parameters, "lr": cfg["head_lr"], "scale": 1.0},
    ], weight_decay=cfg["weight_decay"])
    optimizer.load_state_dict(parent["optimizer"])
    for group, expected_lr in zip(optimizer.param_groups, (cfg["encoder_lr"], cfg["head_lr"])):
        group["lr"] = expected_lr
        if group["weight_decay"] != cfg["weight_decay"] or tuple(group["betas"]) != (0.9, 0.999) or group["eps"] != 1e-8:
            raise AssertionError("optimizer hyperparameters differ from parent")
    return model, optimizer


@torch.no_grad()
def evaluate(model, x, indices, ids, targets, gallery, labels, candidates, records, rows):
    model.eval()
    predictions: list[dict[str, Any]] = []
    total_mse = 0.0
    class_correct = image_correct = 0
    for batch_indices in indices.split(32):
        predicted = model(x[batch_indices])
        per_sample_mse = (predicted - targets[ids[batch_indices]]).square().flatten(1).mean(1)
        total_mse += float(per_sample_mse.sum())
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
                "trial_id": rows[source_index]["trial_id"], "image": rows[source_index]["image"],
                "actual_label": int(rows[source_index]["label_index"]),
                "predicted_image": records[selected]["image_id"], "predicted_label": int(labels[selected]),
                "similarity": float(best[local_index]), "latent_mse": float(per_sample_mse[local_index]),
                "class_correct": int(category_match[local_index]), "image_correct": int(image_match[local_index]),
            })
    return {
        "n": len(indices), "latent_mse": total_mse / len(indices),
        "class_accuracy": class_correct / len(indices), "image_top1": image_correct / len(indices),
        "gallery_size": len(candidates),
    }, predictions


@torch.no_grad()
def clean_train_loss(model, x, train, ids, targets, cfg):
    model.eval()
    total = mse = contrastive = 0.0
    seen = 0
    for batch_indices in train.split(cfg["batch_size"]):
        loss, parts = latent_loss(model(x[batch_indices]), targets[ids[batch_indices]], ids[batch_indices],
                                  temperature=cfg["temperature"], contrastive_weight=cfg["contrastive_weight"])
        n = len(batch_indices)
        total += float(loss) * n
        mse += float(parts["latent_mse"]) * n
        contrastive += float(parts["contrastive"]) * n
        seen += n
    return {"n": seen, "loss": total / seen, "latent_mse": mse / seen, "contrastive": contrastive / seen}


def checkpoint_payload(model, optimizer, arm, epoch, history, contract, aug_generator, cfg):
    return {
        "model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
        "seed": cfg["seed"], "arm": arm, "history": history, "contract": contract,
        "cursor": {"completed_epoch": epoch, "next_epoch": epoch + 1, "completed_steps": (epoch - 70) * 38},
        "rng": {"torch_cpu": torch.get_rng_state(), "torch_cuda": torch.cuda.get_rng_state_all(),
                "augmentation": None if aug_generator is None else aug_generator.get_state()},
    }


def bootstrap_difference(reference: list[dict[str, Any]], candidate: list[dict[str, Any]], seed: int):
    if [row["trial_id"] for row in reference] != [row["trial_id"] for row in candidate]:
        raise AssertionError("paired bootstrap prediction order differs")
    groups: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(reference):
        groups[int(row["actual_label"])].append(index)
    keys = sorted(groups)
    if len(keys) != 80 or any(len(groups[key]) != 20 for key in keys):
        raise AssertionError("expected 80 image-category clusters of 20 test trials")
    rng = np.random.default_rng(seed)
    arrays = {
        "class_accuracy": np.asarray([row["class_correct"] for row in candidate]) - np.asarray([row["class_correct"] for row in reference]),
        "image_top1": np.asarray([row["image_correct"] for row in candidate]) - np.asarray([row["image_correct"] for row in reference]),
        "latent_mse": np.asarray([row["latent_mse"] for row in candidate]) - np.asarray([row["latent_mse"] for row in reference]),
    }
    samples = {key: np.empty(2000) for key in arrays}
    members = [np.asarray(groups[key]) for key in keys]
    for replicate in range(2000):
        chosen = rng.integers(0, 80, size=80)
        index = np.concatenate([members[value] for value in chosen])
        for key, values in arrays.items():
            samples[key][replicate] = values[index].mean()
    return {key: {"observed_difference": float(values.mean()), "percentile_95ci": np.quantile(samples[key], [0.025, 0.975]).tolist()}
            for key, values in arrays.items()}


def self_test() -> None:
    augmentation = augmentation_self_test()
    coords = torch.randn(62, 3)
    model = d066.ImageDecoder(d064.MaskedFrequency(coords, 17), 17)
    model.eval()
    with torch.no_grad():
        output = model(torch.randn(2, 496, 4))
    if output.shape != (2, 256, 1024) or len(d064.order(17, 71, 2400).unique()) != 2400:
        raise AssertionError("D078 model/sampler smoke test failed")
    print(json.dumps({"self_test": "PASS", "augmentation": augmentation["status"], "model_output": list(output.shape),
                      "arms": list(ARMS), "sampler_coverage": 2400}), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()
    self_test()
    if not args.train:
        return 0
    cfg = json.loads(CFG_PATH.read_text(encoding="utf-8"))
    ensure_trial_manifest()
    if cfg["arms"] != list(ARMS) or (cfg["start_epoch"], cfg["end_epoch"], cfg["additional_epochs"]) != (70, 120, 50):
        raise AssertionError("D078 fixed arm/epoch contract differs")
    report = ROOT / cfg["report_dir"]
    report.mkdir(parents=True, exist_ok=True)
    gpu = guard(cfg)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    parent = load_parent(cfg)
    sensitive_path = ROOT / cfg["identity_artifact_dir"] / "sensitive_mask.npy"
    identity_complete = ROOT / cfg["identity_artifact_dir"] / "completed.json"
    if not sensitive_path.exists() or not identity_complete.exists():
        raise RuntimeError("completed leak-excluded identity fit is required")
    sensitive_mask = np.load(sensitive_path, allow_pickle=False)
    if sensitive_mask.shape != (62, 32) or sensitive_mask.dtype != np.bool_ or int(sensitive_mask.sum()) != 248:
        raise AssertionError("frozen sensitive mask differs")
    sensitive = torch.from_numpy(np.flatnonzero(sensitive_mask.reshape(-1))).long().cuda()

    contract = {
        "config": sha(CFG_PATH), "runner": sha(Path(__file__)),
        "augmentation": sha(ROOT / "src/d078_augmentation.py"),
        "d064_runner": sha(ROOT / "scripts/run_d064_mask_ratio.py"),
        "d066_runner": sha(ROOT / "scripts/run_d066_latent_finetune.py"),
        "latent_module": sha(PROJECT_ROOT / "src/eegdecoding/rae_latent_alignment.py"),
        "manifest": sha(ROOT / cfg["manifest"]), "normalization": sha(ROOT / cfg["normalization"]),
        "parent": cfg["parent_checkpoint_sha256"], "identity_summary": sha(identity_complete),
        "sensitive_mask": sha(sensitive_path),
    }
    contract_path = report / "contract.json"
    if contract_path.exists() and json.loads(contract_path.read_text(encoding="utf-8")) != contract:
        raise RuntimeError("D078 run contract changed")
    atomic_json(contract_path, contract)

    emit(report, "loading_resident_data", gpu=gpu)
    x, train, test, coords, rows = prepare_data(cfg, parent, report)
    targets, gallery, ids, labels, image_records, target_contract = load_targets(cfg, rows)
    x, train, test, coords = (value.cuda() for value in (x, train, test, coords))
    candidates = ids[test]
    if len(candidates.unique()) != 1600:
        raise AssertionError("test gallery IDs differ")
    resident = {
        "features_device": str(x.device), "targets_device": str(targets.device), "gallery_device": str(gallery.device),
        "features_shape": list(x.shape), "targets_shape": list(targets.shape), "gallery_shape": list(gallery.shape),
        "allocated_gib": torch.cuda.memory_allocated() / 2**30, "reserved_gib": torch.cuda.memory_reserved() / 2**30,
        "device": torch.cuda.get_device_name(0), "gpu_uuid": cfg["gpu_uuid"],
    }
    atomic_json(report / "resident_data.json", resident)
    emit(report, "resident_data_ready", **resident)

    expected_parent = {"class_accuracy": 0.37125, "image_top1": 0.018125, "latent_mse": 0.9566890954971313}
    if (report / "parent_result.json").exists() and (report / "parent_predictions.json").exists():
        parent_record = json.loads((report / "parent_result.json").read_text(encoding="utf-8"))
        if parent_record.get("checkpoint_sha256") != cfg["parent_checkpoint_sha256"]:
            raise RuntimeError("stale cached parent evaluation")
        parent_metrics = parent_record["metrics"]
        parent_predictions = json.loads((report / "parent_predictions.json").read_text(encoding="utf-8"))
    else:
        parent_model, parent_optimizer = build_model_optimizer(coords, parent, cfg)
        parent_metrics, parent_predictions = evaluate(parent_model, x, test, ids, targets, gallery, labels, candidates, image_records, rows)
        atomic_json(report / "parent_predictions.json", parent_predictions)
        atomic_json(report / "parent_result.json", {"epoch": 70, "metrics": parent_metrics, "checkpoint_sha256": cfg["parent_checkpoint_sha256"]})
        del parent_model, parent_optimizer
        torch.cuda.empty_cache()
    if any(abs(parent_metrics[key] - value) > 2e-6 for key, value in expected_parent.items()):
        raise AssertionError(f"parent endpoint reproduction differs: {parent_metrics}")

    results: list[dict[str, Any]] = []
    prediction_sets: dict[str, list[dict[str, Any]]] = {"parent": parent_predictions}
    for arm_index, arm in enumerate(ARMS):
        if STOP or (ROOT / ".runtime/d078_stop_requested").exists():
            return 76
        out = ROOT / cfg["run_dir"] / arm
        out.mkdir(parents=True, exist_ok=True)
        if (out / "result.json").exists():
            result = json.loads((out / "result.json").read_text(encoding="utf-8"))
            if result["contract"] != contract:
                raise RuntimeError(f"stale completed arm: {arm}")
            results.append(result)
            prediction_sets[arm] = json.loads((out / "test_predictions.json").read_text(encoding="utf-8"))
            continue

        model, optimizer = build_model_optimizer(coords, parent, cfg)
        start_encoder = thash(model.encoder.projection.weight)
        start_head = thash(model.projector[-1].weight)
        augmentation_generator = None
        if arm != "A0":
            augmentation_generator = torch.Generator(device="cuda")
            augmentation_generator.manual_seed(int(cfg["augmentation_seed"]) + arm_index * 100000)
        start_epoch = 70
        history: list[dict[str, Any]] = []
        checkpoints = sorted(out.glob("epoch_*.pt"))
        if checkpoints:
            saved = torch.load(checkpoints[-1], map_location="cuda", weights_only=False)
            if saved["contract"] != contract or saved["arm"] != arm or saved["seed"] != cfg["seed"]:
                raise RuntimeError(f"stale checkpoint: {checkpoints[-1]}")
            model.load_state_dict(saved["model"], strict=True)
            optimizer.load_state_dict(saved["optimizer"])
            start_epoch = int(saved["epoch"])
            history = saved["history"]
            if augmentation_generator is not None:
                augmentation_generator.set_state(saved["rng"]["augmentation"].cpu())
        for group, expected_lr in zip(optimizer.param_groups, (cfg["encoder_lr"], cfg["head_lr"])):
            group["lr"] = expected_lr

        for epoch in range(start_epoch + 1, 121):
            tick = time.perf_counter()
            torch.manual_seed(int(cfg["seed"]) * 10000 + epoch)
            model.train()
            permutation = d064.order(int(cfg["seed"]), epoch, 2400).cuda()
            total = mse_sum = contrastive_sum = 0.0
            seen = steps = 0
            first_batch_loss = None
            for batch_indices in train[permutation].split(int(cfg["batch_size"])):
                optimizer.zero_grad(set_to_none=True)
                clean = x[batch_indices]
                augmented = apply_augmentation(clean, arm, sensitive, augmentation_generator)
                loss, parts = latent_loss(model(augmented), targets[ids[batch_indices]], ids[batch_indices],
                                          temperature=cfg["temperature"], contrastive_weight=cfg["contrastive_weight"])
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"nonfinite loss in {arm} epoch {epoch}")
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"], error_if_nonfinite=True)
                optimizer.step()
                n = len(batch_indices)
                if first_batch_loss is None:
                    first_batch_loss = float(loss.detach())
                total += float(loss.detach()) * n
                mse_sum += float(parts["latent_mse"]) * n
                contrastive_sum += float(parts["contrastive"]) * n
                seen += n
                steps += 1
            torch.cuda.synchronize()
            if seen != 2400 or steps != 38 or len(permutation.unique()) != 2400:
                raise AssertionError("epoch coverage/step contract differs")
            record = {
                "arm": arm, "epoch": epoch, "loss": total / seen, "latent_mse": mse_sum / seen,
                "contrastive": contrastive_sum / seen, "first_batch_loss": first_batch_loss,
                "coverage": seen, "steps": steps, "sampler_hash": thash(permutation),
                "head_lr": cfg["head_lr"], "encoder_lr": cfg["encoder_lr"],
                "seconds": time.perf_counter() - tick, "allocated_gib": torch.cuda.memory_allocated() / 2**30,
                "reserved_gib": torch.cuda.memory_reserved() / 2**30,
            }
            history.append(record)
            atomic_json(out / "epochs.json", history)
            emit(report, "training", **record)
            if epoch in cfg["checkpoint_epochs"]:
                atomic_save(out / f"epoch_{epoch:03d}.pt", checkpoint_payload(
                    model, optimizer, arm, epoch, history, contract, augmentation_generator, cfg
                ))
                if STOP or (ROOT / ".runtime/d078_stop_requested").exists():
                    return 76

        expected_checkpoints = [out / f"epoch_{epoch:03d}.pt" for epoch in cfg["checkpoint_epochs"]]
        if not all(path.exists() for path in expected_checkpoints) or len(list(out.glob("epoch_*.pt"))) != 5:
            raise AssertionError(f"{arm}: expected exactly five D078 checkpoints")
        end_encoder = thash(model.encoder.projection.weight)
        end_head = thash(model.projector[-1].weight)
        if end_encoder == start_encoder or end_head == start_head:
            raise AssertionError(f"{arm}: encoder/head weights did not update")
        emit(report, "final_evaluation", arm=arm, epoch=120)
        test_metrics, predictions = evaluate(model, x, test, ids, targets, gallery, labels, candidates, image_records, rows)
        train_loss = clean_train_loss(model, x, train, ids, targets, cfg)
        atomic_json(out / "test_predictions.json", predictions)
        result = {
            "arm": arm, "seed": cfg["seed"], "subject": 0, "parent_epoch": 70, "epoch": 120,
            "additional_epochs": 50, "test": test_metrics, "clean_train_loss": train_loss,
            "contract": contract, "checkpoint_count": 5,
            "training_seconds": sum(record["seconds"] for record in history),
            "encoder_start_sha256": start_encoder, "encoder_end_sha256": end_encoder,
            "head_start_sha256": start_head, "head_end_sha256": end_head,
            "augmentation": cfg["augmentation"][arm],
            "readout": "image-assisted nearest latent among fixed 1600 subject0 test images; historical exploratory test",
        }
        atomic_json(out / "result.json", result)
        results.append(result)
        prediction_sets[arm] = predictions
        atomic_json(report / "completed_arms.json", results)
        del model, optimizer
        torch.cuda.empty_cache()

    comparisons = {
        "A0_minus_parent": bootstrap_difference(prediction_sets["parent"], prediction_sets["A0"], 78800),
        "A1_minus_A0": bootstrap_difference(prediction_sets["A0"], prediction_sets["A1"], 78801),
        "A2_minus_A0": bootstrap_difference(prediction_sets["A0"], prediction_sets["A2"], 78802),
        "A3_minus_A0": bootstrap_difference(prediction_sets["A0"], prediction_sets["A3"], 78803),
        "unit": "80 image-category clusters; paired fixed-test prediction differences; 2000 bootstrap replicates",
    }
    atomic_json(report / "comparisons.json", comparisons)
    summary = {"status": "complete", "parent": parent_metrics, "results": results, "comparisons": comparisons,
               "resident_data": resident, "test_evaluations_per_arm": 1}
    atomic_json(report / "summary.json", summary)
    atomic_json(ROOT / cfg["run_dir"] / "completed.json", {"status": "complete", "arms": list(ARMS), "contract": contract})
    emit(report, "completed", arms=list(ARMS), results=[{"arm": result["arm"], **result["test"]} for result in results])
    return 0


def stop(*_: Any) -> None:
    global STOP
    STOP = True


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    raise SystemExit(main())
