"""Deterministic balanced cursors and exact D081 40/20/10/30 batches."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Hashable, Iterable, Sequence

import numpy as np
import torch


SOURCE_COUNTS = {
    "imagenet_visual": 40,
    "things_matched_visual": 20,
    "things_other_visual": 10,
    "things_rest": 15,
    "openneuro_rest": 15,
}
EFFECTIVE_BATCH = sum(SOURCE_COUNTS.values())


def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


class BalancedGroupCursor:
    """Round-robin groups, shuffled without replacement within every group."""

    def __init__(self, indices: Sequence[int], group_keys: Sequence[Hashable], seed: int, name: str):
        if len(indices) != len(group_keys) or len(indices) == 0:
            raise ValueError(f"{name}: indices/group keys must be nonempty and aligned")
        grouped: dict[str, list[int]] = defaultdict(list)
        for index, key in zip(indices, group_keys):
            grouped[repr(key)].append(int(index))
        self.name = str(name)
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)
        self.groups = sorted(grouped)
        self.base = {key: np.asarray(grouped[key], dtype=np.int64) for key in self.groups}
        self.orders = {key: self.rng.permutation(values) for key, values in self.base.items()}
        self.positions = {key: 0 for key in self.groups}
        self.cycles = {key: 0 for key in self.groups}
        self.group_position = 0
        self.draws = 0
        self.unique_seen: set[int] = set()

    def _one(self, key: str) -> int:
        position = self.positions[key]
        if position >= len(self.orders[key]):
            self.orders[key] = self.rng.permutation(self.base[key])
            self.positions[key] = 0
            self.cycles[key] += 1
            position = 0
        value = int(self.orders[key][position])
        self.positions[key] = position + 1
        self.draws += 1
        self.unique_seen.add(value)
        return value

    def take(self, count: int) -> np.ndarray:
        if count <= 0:
            raise ValueError("take count must be positive")
        output = np.empty(count, dtype=np.int64)
        for index in range(count):
            key = self.groups[self.group_position]
            self.group_position = (self.group_position + 1) % len(self.groups)
            output[index] = self._one(key)
        return output

    def audit(self) -> dict[str, Any]:
        pool = sum(len(values) for values in self.base.values())
        return {
            "name": self.name, "groups": len(self.groups), "pool_trials": pool,
            "draws": self.draws, "unique_trials_seen": len(self.unique_seen),
            "unique_coverage": len(self.unique_seen) / pool,
            "max_group_cycle": max(self.cycles.values()), "min_group_cycle": min(self.cycles.values()),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "seed": self.seed, "groups": self.groups,
            "base_hashes": {key: array_sha256(value) for key, value in self.base.items()},
            "orders": {key: value.copy() for key, value in self.orders.items()},
            "positions": dict(self.positions), "cycles": dict(self.cycles),
            "group_position": self.group_position, "draws": self.draws,
            "unique_seen": np.asarray(sorted(self.unique_seen), dtype=np.int64),
            "rng_state": self.rng.bit_generator.state,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state["name"] != self.name or int(state["seed"]) != self.seed or list(state["groups"]) != self.groups:
            raise AssertionError(f"{self.name}: cursor identity differs")
        hashes = {key: array_sha256(value) for key, value in self.base.items()}
        if state["base_hashes"] != hashes:
            raise AssertionError(f"{self.name}: cursor pool differs")
        self.orders = {key: np.asarray(value, dtype=np.int64) for key, value in state["orders"].items()}
        self.positions = {key: int(value) for key, value in state["positions"].items()}
        self.cycles = {key: int(value) for key, value in state["cycles"].items()}
        self.group_position = int(state["group_position"])
        self.draws = int(state["draws"])
        self.unique_seen = set(np.asarray(state["unique_seen"], dtype=np.int64).tolist())
        self.rng.bit_generator.state = state["rng_state"]


@dataclass
class D081Batch:
    imagenet_visual: np.ndarray
    things_matched_visual: np.ndarray
    things_other_visual: np.ndarray
    things_rest: np.ndarray
    openneuro_rest: np.ndarray

    def audit(self) -> dict[str, int]:
        value = {name: len(getattr(self, name)) for name in SOURCE_COUNTS}
        if value != SOURCE_COUNTS or sum(value.values()) != EFFECTIVE_BATCH:
            raise AssertionError(f"D081 source ratio differs: {value}")
        return value


class D081Sampler:
    def __init__(self, cursors: dict[str, BalancedGroupCursor]):
        if set(cursors) != set(SOURCE_COUNTS):
            raise ValueError(f"D081 sampler sources differ: {set(cursors)}")
        self.cursors = cursors
        self.steps = 0
        self.cumulative = {key: 0 for key in SOURCE_COUNTS}

    @property
    def steps_per_round(self) -> int:
        pool = sum(len(value) for value in self.cursors["imagenet_visual"].base.values())
        return math.ceil(pool / SOURCE_COUNTS["imagenet_visual"])

    def next(self) -> D081Batch:
        batch = D081Batch(**{
            name: self.cursors[name].take(count) for name, count in SOURCE_COUNTS.items()
        })
        counts = batch.audit()
        self.steps += 1
        for name, count in counts.items():
            self.cumulative[name] += count
        expected = {name: count * self.steps for name, count in SOURCE_COUNTS.items()}
        if self.cumulative != expected:
            raise AssertionError("D081 cumulative source ratio drifted")
        return batch

    def audit(self) -> dict[str, Any]:
        return {
            "steps": self.steps, "cumulative": dict(self.cumulative),
            "ratio_assertion": self.cumulative == {name: count * self.steps for name, count in SOURCE_COUNTS.items()},
            "sources": {name: cursor.audit() for name, cursor in self.cursors.items()},
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "steps": self.steps, "cumulative": dict(self.cumulative),
            "cursors": {name: cursor.state_dict() for name, cursor in self.cursors.items()},
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.steps = int(state["steps"])
        self.cumulative = {name: int(value) for name, value in state["cumulative"].items()}
        for name, cursor_state in state["cursors"].items():
            self.cursors[name].load_state_dict(cursor_state)
        self.audit()


class EOECRestCursor:
    """External-rest cursor with the exact repeating 3:1 EO/EC schedule."""

    schedule = ((11, 4), (11, 4), (11, 4), (12, 3))

    def __init__(self, eyes_open: BalancedGroupCursor, eyes_closed: BalancedGroupCursor, name: str = "openneuro_rest"):
        self.eyes_open = eyes_open
        self.eyes_closed = eyes_closed
        self.name = name
        self.steps = 0
        self.draws = 0

    @property
    def base(self) -> dict[str, np.ndarray]:
        return {**{f"eo/{key}": value for key, value in self.eyes_open.base.items()},
                **{f"ec/{key}": value for key, value in self.eyes_closed.base.items()}}

    def take(self, count: int) -> np.ndarray:
        if count != 15:
            raise ValueError("D081 OpenNeuro cursor must draw exactly 15 windows/step")
        eo_count, ec_count = self.schedule[self.steps % len(self.schedule)]
        output = np.concatenate((self.eyes_open.take(eo_count), self.eyes_closed.take(ec_count)))
        # Shuffle only the within-batch order deterministically; membership and
        # the 45/15 four-step condition count are unchanged.
        rng = np.random.default_rng(810000 + self.steps)
        output = output[rng.permutation(len(output))]
        self.steps += 1
        self.draws += len(output)
        return output

    def audit(self) -> dict[str, Any]:
        return {"name": self.name, "steps": self.steps, "draws": self.draws,
                "four_step_eo_ec": [45, 15], "eyes_open": self.eyes_open.audit(),
                "eyes_closed": self.eyes_closed.audit()}

    def state_dict(self) -> dict[str, Any]:
        return {"name": self.name, "steps": self.steps, "draws": self.draws,
                "eyes_open": self.eyes_open.state_dict(), "eyes_closed": self.eyes_closed.state_dict()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state["name"] != self.name:
            raise AssertionError("EO/EC cursor identity differs")
        self.steps = int(state["steps"])
        self.draws = int(state["draws"])
        self.eyes_open.load_state_dict(state["eyes_open"])
        self.eyes_closed.load_state_dict(state["eyes_closed"])


def synchronous_temporal_mask(batch: int, ratio: float, generator: torch.Generator,
                              device: torch.device | str) -> torch.Tensor:
    """Stochastic rounding of 8*r, then uniform positions without replacement."""
    if not 0.0 <= ratio <= 1.0:
        raise ValueError("mask ratio must be in [0,1]")
    exact = 8.0 * ratio
    lower = math.floor(exact)
    fraction = exact - lower
    counts = torch.full((batch,), lower, device=device, dtype=torch.long)
    if fraction:
        counts += (torch.rand(batch, generator=generator, device=device) < fraction).long()
    counts.clamp_(1, 4)
    noise = torch.rand((batch, 8), generator=generator, device=device)
    ranks = noise.argsort(dim=1).argsort(dim=1)
    mask = ranks < counts[:, None]
    if not torch.equal(mask.sum(1), counts):
        raise AssertionError("temporal mask count differs")
    return mask


def sampler_self_test() -> dict[str, Any]:
    cursors = {}
    for source_index, (name, count) in enumerate(SOURCE_COUNTS.items()):
        pool = np.arange(max(2 * count, 120)) + source_index * 1000
        groups = [(int(value) % 3, int(value) % 5) for value in pool]
        cursors[name] = BalancedGroupCursor(pool, groups, seed=17 + source_index, name=name)
    sampler = D081Sampler(cursors)
    for _ in range(7):
        sampler.next()
    state = sampler.state_dict()
    expected = sampler.next()
    restored_cursors = {}
    for source_index, (name, count) in enumerate(SOURCE_COUNTS.items()):
        pool = np.arange(max(2 * count, 120)) + source_index * 1000
        groups = [(int(value) % 3, int(value) % 5) for value in pool]
        restored_cursors[name] = BalancedGroupCursor(pool, groups, seed=17 + source_index, name=name)
    restored = D081Sampler(restored_cursors)
    restored.load_state_dict(state)
    actual = restored.next()
    if any(not np.array_equal(getattr(expected, name), getattr(actual, name)) for name in SOURCE_COUNTS):
        raise AssertionError("sampler resume sequence differs")
    generator = torch.Generator().manual_seed(17)
    mask = synchronous_temporal_mask(100, 0.37, generator, "cpu")
    if mask.shape != (100, 8) or int(mask.sum(1).min()) < 1 or int(mask.sum(1).max()) > 4:
        raise AssertionError("mask contract differs")
    return {"status": "PASS", "source_counts": SOURCE_COUNTS, "effective_batch": EFFECTIVE_BATCH,
            "resume_exact": True, "mask_shape": list(mask.shape),
            "mask_min_max": [int(mask.sum(1).min()), int(mask.sum(1).max())]}


__all__ = ["BalancedGroupCursor", "D081Sampler", "D081Batch", "SOURCE_COUNTS", "EFFECTIVE_BATCH",
           "synchronous_temporal_mask", "sampler_self_test"]
