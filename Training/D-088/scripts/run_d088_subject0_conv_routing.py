"""Prepare, validate, and run the D088 four-way convolution-routing experiment."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
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
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from d088_dual_attention import (  # noqa: E402
    CHANNELS,
    PATCHES,
    ROUTES,
    D088Classifier,
    D088JEPA,
    parameter_counts,
    periodic_hann_log_psd,
)
from eegdecoding.subject_data import load_subject_data  # noqa: E402

OUTPUT = ROOT / "runs/d088-subject0-conv-routing"
REPORT = ROOT / "reports/d088_subject0_conv_routing"
STATS = REPORT / "subject0_train_stats.npz"
SPLIT = REPORT / "subject0_split.npz"
SEED = 17


def stable_key(image_id: str) -> str:
    return hashlib.sha256(f"D088-seed{SEED}:{image_id}".encode()).hexdigest()


def prepare() -> dict:
    REPORT.mkdir(parents=True, exist_ok=True)
    data = load_subject_data({
        "archive": "EEG-ImageNet_1.pth",
        "subject": 0,
        "crop": [40, 440],
        "development_split": {"hash_prefix": f"D088-seed{SEED}:"},
    })
    waveform = data["waveform"]
    labels = data["labels"].numpy()
    train = data["fit"].numpy().astype(np.int32)
    dev = data["validation"].numpy().astype(np.int32)
    test = data["test"].numpy().astype(np.int32)
    np.savez(SPLIT, train=train, dev=dev, test=test)

    sum_value = np.zeros(CHANNELS, dtype=np.float64)
    sum_square = np.zeros(CHANNELS, dtype=np.float64)
    count = 0
    for start in range(0, len(train), 64):
        value = waveform[train[start:start + 64]].double().numpy()
        sum_value += value.sum(axis=(0, 2))
        sum_square += np.square(value).sum(axis=(0, 2))
        count += value.shape[0] * value.shape[2]
    wave_mean = sum_value / count
    wave_var = np.maximum(sum_square / count - np.square(wave_mean), 1e-16)
    wave_scale = np.sqrt(wave_var)

    frequency_sum = np.zeros((CHANNELS, 32), dtype=np.float64)
    frequency_square = np.zeros((CHANNELS, 32), dtype=np.float64)
    frequency_count = 0
    for start in range(0, len(train), 32):
        value = waveform[train[start:start + 32]].numpy()
        value = (value - wave_mean[None, :, None]) / wave_scale[None, :, None]
        with torch.no_grad():
            spectrum = periodic_hann_log_psd(torch.from_numpy(value)).numpy().astype(np.float64)
        frequency_sum += spectrum.sum(axis=0)
        frequency_square += np.square(spectrum).sum(axis=0)
        frequency_count += len(value)
    frequency_mean = frequency_sum / frequency_count
    frequency_var = np.maximum(frequency_square / frequency_count - np.square(frequency_mean), 1e-8)
    frequency_scale = np.sqrt(frequency_var)
    np.savez(
        STATS,
        wave_mean=wave_mean.astype(np.float32),
        wave_scale=wave_scale.astype(np.float32),
        frequency_mean=frequency_mean.astype(np.float32),
        frequency_scale=frequency_scale.astype(np.float32),
    )
    result = {
        "status": "PASS",
        "subject": 0,
        "train": int(len(train)),
        "dev": int(len(dev)),
        "test": int(len(test)),
        "train_per_class": 27,
        "dev_per_class": 3,
        "test_per_class": 20,
        "source": data["source_kind"],
        "test_used_for_selection": False,
    }
    (REPORT / "data_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


class TrialBank:
    def __init__(self):
        data = load_subject_data({"archive": "EEG-ImageNet_1.pth", "subject": 0, "crop": [40, 440]})
        self.waveform = data["waveform"]
        self.labels = data["labels"].numpy().astype(np.int64)
        split = np.load(SPLIT)
        self.indices = {name: split[name].astype(np.int64) for name in split.files}
        stats = np.load(STATS)
        self.wave_mean = stats["wave_mean"]
        self.wave_scale = stats["wave_scale"]
        self.frequency_mean = torch.from_numpy(stats["frequency_mean"])
        self.frequency_scale = torch.from_numpy(stats["frequency_scale"])

    def batch(self, indices: np.ndarray, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        value = self.waveform[indices].numpy()
        value = (value - self.wave_mean[None, :, None]) / self.wave_scale[None, :, None]
        return torch.from_numpy(value).to(device, non_blocking=True), torch.from_numpy(self.labels[indices]).to(device)


def make_mask(size: int, ratio: float, generator: torch.Generator, device: torch.device) -> torch.Tensor:
    exact = min(4.0, max(1.0, PATCHES * ratio))
    lower = int(math.floor(exact))
    fraction = exact - lower
    masks = torch.zeros((size, PATCHES), dtype=torch.bool)
    for row in range(size):
        length = lower + int(torch.rand((), generator=generator).item() < fraction)
        start = int(torch.randint(PATCHES - length + 1, (), generator=generator).item())
        masks[row, start:start + length] = True
    return masks.to(device)


def write_log(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")


def route_trainable(model: D088JEPA, route: str) -> None:
    for encoder in (model.online, model.target):
        for block in encoder.blocks:
            block.time_attention.conv_secondary.requires_grad_(route == "separate_conv" and encoder is model.online)


def smoke(device: torch.device) -> dict:
    if not SPLIT.exists() or not STATS.exists():
        prepare()
    bank = TrialBank()
    index = bank.indices["train"][:1]
    waveform, _ = bank.batch(index, device)
    frequency_mean = bank.frequency_mean.to(device)
    frequency_scale = bank.frequency_scale.to(device)
    results = {}
    for route in ROUTES:
        torch.manual_seed(SEED)
        model = D088JEPA(route, dropout=0.0, activation_checkpointing=True).to(device)
        route_trainable(model, route)
        hidden = torch.tensor([[False, True, True, False, False, False, False, False]], device=device)
        altered = waveform.clone().reshape(1, CHANNELS, PATCHES, -1)
        altered[:, :, hidden[0]] = torch.randn_like(altered[:, :, hidden[0]]) * 100
        altered = altered.reshape_as(waveform)
        model.eval()
        with torch.no_grad():
            a = model.online(waveform, hidden, frequency_mean, frequency_scale)
            b = model.online(altered, hidden, frequency_mean, frequency_scale)
        leak = float((a - b).abs().max())
        if leak > 1e-5:
            raise AssertionError(f"{route}: hidden-content leakage {leak}")
        model.train()
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4)
        optimizer.zero_grad(set_to_none=True)
        loss, parts = model.loss(waveform, hidden, frequency_mean, frequency_scale)
        loss.backward()
        before = model.online.wave_patch[0].weight.detach().clone()
        optimizer.step()
        update = float((model.online.wave_patch[0].weight.detach() - before).abs().max())
        if not math.isfinite(float(loss.detach())) or update <= 0:
            raise AssertionError((route, float(loss), update))
        results[route] = {"loss": float(loss), "cosine": float(parts["cosine"]), "update": update,
                          "leakage_max": leak, "parameters": parameter_counts(model)}
        del model, optimizer
        torch.cuda.empty_cache()
    result = {"status": "PASS", "device": str(device), "routes": results}
    (REPORT / "smoke.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def evaluate(model: D088Classifier, bank: TrialBank, split: str, device: torch.device, microbatch: int) -> dict:
    model.eval()
    correct = np.zeros(80, dtype=np.int64)
    total = np.zeros(80, dtype=np.int64)
    losses = []
    indices = bank.indices[split]
    frequency_mean = bank.frequency_mean.to(device)
    frequency_scale = bank.frequency_scale.to(device)
    with torch.no_grad():
        for start in range(0, len(indices), microbatch):
            waveform, labels = bank.batch(indices[start:start + microbatch], device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(waveform, frequency_mean, frequency_scale)
            losses.append(float(F.cross_entropy(logits.float(), labels)) * len(labels))
            prediction = logits.argmax(dim=-1)
            for label, predicted in zip(labels.cpu().numpy(), prediction.cpu().numpy()):
                total[label] += 1
                correct[label] += int(label == predicted)
    per_class = correct / np.maximum(total, 1)
    return {"loss": float(sum(losses) / len(indices)), "top1": float(correct.sum() / total.sum()),
            "macro_class_top1": float(per_class.mean()), "per_class_top1": per_class.tolist(), "n": int(len(indices))}


def pretrain(route: str, bank: TrialBank, device: torch.device, steps: int, microbatch: int) -> Path:
    run = OUTPUT / route
    run.mkdir(parents=True, exist_ok=True)
    log = run / "pretrain.jsonl"
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED); np.random.seed(SEED); random.seed(SEED)
    # The local 8 GiB GPU has enough headroom for retained activations. Keeping
    # them avoids recomputing all eight online blocks during backward.
    model = D088JEPA(route, activation_checkpointing=False).to(device)
    route_trainable(model, route)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4,
                                  betas=(0.9, 0.999), weight_decay=1e-3)
    rng = np.random.default_rng(SEED + 8800)
    mask_rng = torch.Generator().manual_seed(SEED + 8801)
    order = bank.indices["train"].copy(); rng.shuffle(order); cursor = 0
    frequency_mean = bank.frequency_mean.to(device); frequency_scale = bank.frequency_scale.to(device)
    started = time.perf_counter()
    for step in range(1, steps + 1):
        logical = []
        while len(logical) < 100:
            take = min(100 - len(logical), len(order) - cursor)
            logical.extend(order[cursor:cursor + take].tolist()); cursor += take
            if cursor == len(order):
                rng.shuffle(order); cursor = 0
        logical = np.asarray(logical, dtype=np.int64)
        ratio = 0.15 + 0.35 * min(step, int(steps * 0.75)) / max(1, int(steps * 0.75))
        optimizer.zero_grad(set_to_none=True)
        sum_loss = 0.0; sum_cosine = 0.0
        for start in range(0, len(logical), microbatch):
            ids = logical[start:start + microbatch]
            waveform, _ = bank.batch(ids, device)
            hidden = make_mask(len(ids), ratio, mask_rng, device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss, parts = model.loss(waveform, hidden, frequency_mean, frequency_scale)
            weight = len(ids) / len(logical)
            (loss * weight).backward()
            sum_loss += float(loss) * len(ids); sum_cosine += float(parts["cosine"]) * len(ids)
        grad = float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0))
        warmup = max(1, int(steps * 0.075))
        if step <= warmup:
            lr = 3e-4 * step / warmup
        else:
            phase = (step - warmup) / max(1, steps - warmup)
            lr = 3e-5 + 0.5 * (3e-4 - 3e-5) * (1 + math.cos(math.pi * phase))
        for group in optimizer.param_groups: group["lr"] = lr
        optimizer.step()
        momentum = 0.996 + 0.004 * step / steps
        model.update_target(momentum)
        if step == 1 or step % 100 == 0:
            write_log(log, {"stage": "pretrain", "route": route, "step": step, "steps": steps,
                            "loss": sum_loss / 100, "cosine": sum_cosine / 100, "grad_norm": grad,
                            "lr": lr, "mask_ratio": ratio, "seconds": time.perf_counter() - started,
                            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30})
        # Keep one midpoint recovery checkpoint.  The completed state is written
        # once below as pretrain_final.pt, so do not duplicate the final step.
        if step % 1000 == 0 and step < steps:
            torch.save({"route": route, "step": step, "online": model.online.state_dict(),
                        "target": model.target.state_dict(), "predictor": model.predictor.state_dict(),
                        "optimizer": optimizer.state_dict(), "rng": rng.bit_generator.state,
                        "mask_rng": mask_rng.get_state()}, run / f"pretrain_step{step:05d}.pt")
    path = run / "pretrain_final.pt"
    torch.save({"route": route, "step": steps, "online": model.online.state_dict(),
                "target": model.target.state_dict(), "parameters": parameter_counts(model)}, path)
    del model, optimizer
    torch.cuda.empty_cache()
    return path


def finetune(route: str, checkpoint: Path, bank: TrialBank, device: torch.device, epochs: int, microbatch: int) -> dict:
    run = OUTPUT / route
    log = run / "finetune.jsonl"
    torch.manual_seed(SEED + 1); torch.cuda.manual_seed_all(SEED + 1)
    jepa = D088JEPA(route, activation_checkpointing=False)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    jepa.target.load_state_dict(state["target"])
    encoder = copy.deepcopy(jepa.target); del jepa
    encoder.requires_grad_(True)
    if route != "separate_conv":
        for block in encoder.blocks: block.time_attention.conv_secondary.requires_grad_(False)
    model = D088Classifier(encoder).to(device)
    encoder_parameters = [p for p in model.encoder.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([{"params": encoder_parameters, "lr": 3e-5},
                                   {"params": model.head.parameters(), "lr": 3e-4},
                                   {"params": model.norm.parameters(), "lr": 3e-4}], weight_decay=1e-3)
    rng = np.random.default_rng(SEED + 8802)
    frequency_mean = bank.frequency_mean.to(device); frequency_scale = bank.frequency_scale.to(device)
    best = {"epoch": 0, "top1": -1.0}; best_path = run / "finetune_best.pt"
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        order = bank.indices["train"].copy(); rng.shuffle(order)
        model.train(); train_loss = 0.0; train_correct = 0
        for logical_start in range(0, len(order), 64):
            logical = order[logical_start:logical_start + 64]
            optimizer.zero_grad(set_to_none=True)
            for start in range(0, len(logical), microbatch):
                ids = logical[start:start + microbatch]
                waveform, labels = bank.batch(ids, device)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    logits = model(waveform, frequency_mean, frequency_scale)
                    loss = F.cross_entropy(logits.float(), labels)
                (loss * len(ids) / len(logical)).backward()
                train_loss += float(loss) * len(ids)
                train_correct += int((logits.argmax(-1) == labels).sum())
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            progress = ((epoch - 1) + min(1.0, (logical_start + len(logical)) / len(order))) / epochs
            enc_lr = 3e-6 + 0.5 * (3e-5 - 3e-6) * (1 + math.cos(math.pi * progress))
            head_lr = 3e-5 + 0.5 * (3e-4 - 3e-5) * (1 + math.cos(math.pi * progress))
            optimizer.param_groups[0]["lr"] = enc_lr
            optimizer.param_groups[1]["lr"] = head_lr
            optimizer.param_groups[2]["lr"] = head_lr
            optimizer.step()
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            dev = evaluate(model, bank, "dev", device, microbatch)
            row = {"stage": "finetune", "route": route, "epoch": epoch, "epochs": epochs,
                   "train_loss": train_loss / len(order), "train_top1": train_correct / len(order),
                   "dev": dev, "seconds": time.perf_counter() - started,
                   "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30}
            write_log(log, row)
            if dev["macro_class_top1"] > best["top1"]:
                best = {"epoch": epoch, "top1": dev["macro_class_top1"], "dev": dev}
                torch.save({"route": route, "epoch": epoch, "model": model.state_dict(), "dev": dev}, best_path)
        # Four milestones over 70 epochs are sufficient for trajectory review;
        # finetune_best.pt separately preserves the selected validation model.
        if epoch % 20 == 0 or epoch == epochs:
            torch.save({"route": route, "epoch": epoch, "model": model.state_dict(),
                        "optimizer": optimizer.state_dict()}, run / f"finetune_epoch{epoch:03d}.pt")
    best_state = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(best_state["model"])
    test = evaluate(model, bank, "test", device, microbatch)
    result = {"route": route, "best": best, "test": test, "parameters": parameter_counts(model),
              "elapsed_seconds": time.perf_counter() - started}
    (run / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    del model, optimizer
    torch.cuda.empty_cache()
    return result


def run_all(device: torch.device, steps: int, epochs: int, microbatch: int) -> dict:
    if not SPLIT.exists() or not STATS.exists(): prepare()
    bank = TrialBank()
    results = {}
    REPORT.mkdir(parents=True, exist_ok=True); OUTPUT.mkdir(parents=True, exist_ok=True)
    status_path = REPORT / "status.json"
    for route in ROUTES:
        status_path.write_text(json.dumps({"status": "running", "route": route, "stage": "pretrain",
                                           "pid": os.getpid(), "steps": steps, "epochs": epochs}, indent=2), encoding="utf-8")
        checkpoint = pretrain(route, bank, device, steps, microbatch)
        status_path.write_text(json.dumps({"status": "running", "route": route, "stage": "finetune",
                                           "pid": os.getpid(), "steps": steps, "epochs": epochs}, indent=2), encoding="utf-8")
        results[route] = finetune(route, checkpoint, bank, device, epochs, microbatch)
    summary = {"status": "complete", "subject": 0, "pretrain_steps": steps, "finetune_epochs": epochs,
               "device": str(device), "results": results}
    (REPORT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    status_path.write_text(json.dumps({"status": "complete", "pid": os.getpid()}, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--run-all", action="store_true")
    parser.add_argument("--pretrain-steps", type=int, default=2000)
    parser.add_argument("--finetune-epochs", type=int, default=70)
    parser.add_argument("--microbatch", type=int, default=4)
    args = parser.parse_args()
    if not any((args.prepare, args.smoke, args.run_all)):
        parser.error("select --prepare, --smoke, or --run-all")
    if args.prepare: print(json.dumps(prepare(), indent=2))
    if args.smoke:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(json.dumps(smoke(device), indent=2))
    if args.run_all:
        if not torch.cuda.is_available(): raise RuntimeError("formal D088 training requires CUDA")
        print(json.dumps(run_all(torch.device("cuda"), args.pretrain_steps, args.finetune_epochs, args.microbatch), indent=2))


if __name__ == "__main__":
    main()
