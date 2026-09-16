"""D080 helpers for random-state sharing and exact step-based learning rates."""

from __future__ import annotations

import hashlib
import math
from typing import Any

import torch
from torch import nn

from d079_architecture import EarlyFusionFrequency, NoCoordinateFrequency
from eegdecoding.rae_latent_alignment import QueryBlock, grid_encoding


ARMS = ("B0", "B1", "B2")


class CoordinateFrequency(nn.Module):
    """Fresh D064-shape encoder; no trained state is accepted by this class."""

    token_count = 496

    def __init__(self, coordinates: torch.Tensor, seed: int):
        super().__init__()
        torch.manual_seed(seed)
        self.register_buffer("coordinates", coordinates.clone())
        self.projection = nn.Linear(4, 128)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 128))
        nn.init.normal_(self.mask_token, std=0.02)
        self.electrode = nn.Embedding(62, 128)
        self.position = nn.Linear(3, 128)
        self.frequency_position = nn.Embedding(8, 128)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(128, 4, 384, 0.1, activation="gelu", batch_first=True)
            for _ in range(2)
        ])
        self.output = nn.Linear(128, 4)

    def tokens(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1:] != (496, 4):
            raise ValueError("D080 B0 input must be [B,496,4]")
        h = self.projection(x)
        if mask is not None:
            h = torch.where(mask[..., None], self.mask_token, h)
        position = (
            (self.electrode.weight + self.position(self.coordinates))[:, None, :]
            + self.frequency_position.weight[None, :, :]
        ).reshape(496, 128)
        return h + position

    def encode(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.tokens(x, mask)
        for layer in self.layers:
            h = layer(h)
        return h

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.output(self.encode(x, mask))


class ScratchImageDecoder(nn.Module):
    """Fresh D066-shape image-latent head with its unused masked head frozen."""

    def __init__(self, encoder: nn.Module, seed: int):
        super().__init__()
        self.encoder = encoder
        self.encoder.output.requires_grad_(False)
        self.encoder.mask_token.requires_grad_(False)
        torch.manual_seed(seed + 66000)
        self.memory_norm = nn.LayerNorm(128)
        self.queries = nn.Parameter(torch.empty(256, 128))
        nn.init.trunc_normal_(self.queries, std=0.02)
        self.register_buffer("grid", grid_encoding(16, 128))
        self.blocks = nn.ModuleList([QueryBlock(128, 4, 0.1) for _ in range(2)])
        self.projector = nn.Sequential(
            nn.LayerNorm(128), nn.Linear(128, 512), nn.GELU(), nn.Dropout(0.1), nn.Linear(512, 1024)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        memory = self.memory_norm(self.encoder.encode(x))
        query = (self.queries + self.grid)[None].expand(len(x), -1, -1)
        for block in self.blocks:
            query = block(query, memory)
        return self.projector(query)


def build_random_family(coordinates: torch.Tensor, seed: int) -> tuple[dict[str, ScratchImageDecoder], dict[str, list[str]]]:
    """Construct one fresh B0 source, then copy only its compatible random state."""
    source = ScratchImageDecoder(CoordinateFrequency(coordinates, seed), seed)
    models: dict[str, ScratchImageDecoder] = {
        "B0": source,
        "B1": ScratchImageDecoder(NoCoordinateFrequency(seed), seed),
        "B2": ScratchImageDecoder(EarlyFusionFrequency(seed), seed),
    }
    copied = {"B0": sorted(source.state_dict())}
    copied["B1"] = copy_random_shared_state(source, models["B1"])
    copied["B2"] = copy_random_shared_state(source, models["B2"])
    shared_initialization_evidence(models)
    return models, copied


def tensor_sha(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def model_state_sha(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def shared_state_names(models: dict[str, nn.Module]) -> list[str]:
    common = set.intersection(*(set(model.state_dict()) for model in models.values()))
    return sorted(
        name for name in common
        if "coordinates" not in name
        and not ("position" in name and "frequency_position" not in name)
        and name != "encoder.fusion"
    )


def copy_random_shared_state(source: nn.Module, target: nn.Module) -> list[str]:
    source_state = source.state_dict()
    target_state = target.state_dict()
    copied = []
    for name, value in source_state.items():
        if name in target_state and target_state[name].shape == value.shape:
            if "coordinates" in name or ("position" in name and "frequency_position" not in name):
                continue
            target_state[name].copy_(value)
            copied.append(name)
    target.load_state_dict(target_state, strict=True)
    return sorted(copied)


def shared_initialization_evidence(models: dict[str, nn.Module]) -> dict[str, Any]:
    names = shared_state_names(models)
    per_arm = {
        arm: {
            name: tensor_sha(model.state_dict()[name])
            for name in names
        }
        for arm, model in models.items()
    }
    reference = per_arm["B0"]
    if any(hashes != reference for hashes in per_arm.values()):
        raise AssertionError("D080 shared random state differs across arms")
    digest = hashlib.sha256()
    for name, value in sorted(reference.items()):
        digest.update(name.encode())
        digest.update(value.encode())
    parameter_maps = {arm: dict(model.named_parameters()) for arm, model in models.items()}
    shared_learnable_names = sorted(
        name for name in names
        if all(name in parameter_maps[arm] and parameter_maps[arm][name].requires_grad for arm in models)
    )
    learnable_digest = hashlib.sha256()
    for name in shared_learnable_names:
        learnable_digest.update(name.encode())
        learnable_digest.update(reference[name].encode())
    return {
        "source": "random_initialization",
        "seed": 17,
        "shared_state_entry_count": len(names),
        "shared_numel": sum(models["B0"].state_dict()[name].numel() for name in names),
        "shared_combined_sha256": digest.hexdigest(),
        "all_arm_hash_maps_identical": True,
        "shared_learnable_parameter_count": len(shared_learnable_names),
        "shared_learnable_numel": sum(models["B0"].state_dict()[name].numel() for name in shared_learnable_names),
        "shared_learnable_combined_sha256": learnable_digest.hexdigest(),
        "shared_learnable_names": shared_learnable_names,
        "state_names": names,
        "hashes": reference,
    }


class ExactWarmupCosine:
    """Set LR before each optimizer update using the D080 1-based step contract."""

    def __init__(self, optimizer: torch.optim.Optimizer, peak_lr: float, min_lr: float,
                 warmup_steps: int, total_steps: int):
        self.optimizer = optimizer
        self.peak_lr = float(peak_lr)
        self.min_lr = float(min_lr)
        self.warmup_steps = int(warmup_steps)
        self.total_steps = int(total_steps)
        self.completed_steps = 0
        for group in self.optimizer.param_groups:
            group["lr"] = 0.0

    def lr_at(self, step: int) -> float:
        if not 1 <= step <= self.total_steps:
            raise ValueError(f"D080 scheduler step out of range: {step}")
        if step <= self.warmup_steps:
            return self.peak_lr * step / self.warmup_steps
        progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        return self.min_lr + 0.5 * (self.peak_lr - self.min_lr) * (1.0 + math.cos(math.pi * progress))

    def step_before_optimizer(self) -> float:
        step = self.completed_steps + 1
        learning_rate = self.lr_at(step)
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate
        self.completed_steps = step
        return learning_rate

    def state_dict(self) -> dict[str, Any]:
        return {
            "peak_lr": self.peak_lr,
            "min_lr": self.min_lr,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "completed_steps": self.completed_steps,
            "last_lr": 0.0 if self.completed_steps == 0 else self.lr_at(self.completed_steps),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        expected = {key: self.state_dict()[key] for key in ("peak_lr", "min_lr", "warmup_steps", "total_steps")}
        if any(state[key] != value for key, value in expected.items()):
            raise AssertionError("D080 scheduler contract changed on resume")
        self.completed_steps = int(state["completed_steps"])
        learning_rate = 0.0 if self.completed_steps == 0 else self.lr_at(self.completed_steps)
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate


def scheduler_self_test() -> dict[str, Any]:
    parameter = nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.AdamW([parameter], lr=1.0)
    scheduler = ExactWarmupCosine(optimizer, 3e-4, 3e-5, 190, 2660)
    values = {step: scheduler.lr_at(step) for step in (1, 189, 190, 191, 2659, 2660)}
    if values[1] != 3e-4 / 190 or values[190] != 3e-4 or abs(values[2660] - 3e-5) > 1e-18:
        raise AssertionError(f"D080 scheduler endpoints differ: {values}")
    previous = -1.0
    for step in range(1, 191):
        current = scheduler.lr_at(step)
        if current <= previous:
            raise AssertionError("D080 warmup is not strictly increasing")
        previous = current
    previous = scheduler.lr_at(190)
    for step in range(191, 2661):
        current = scheduler.lr_at(step)
        if current > previous:
            raise AssertionError("D080 cosine decay is not monotone")
        previous = current
    return {"status": "PASS", "checkpoints": values}
