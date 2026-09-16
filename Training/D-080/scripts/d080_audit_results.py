"""Audit D080 fixed evaluations, predictions, checkpoints, and fairness invariants."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "config/d080_from_scratch_no_coord.json"
ARMS = ("B0", "B1", "B2")
EPOCHS = (0, 10, 20, 30, 40, 50, 60, 70)
CHECKPOINT_EPOCHS = EPOCHS[1:]


def model_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def close(left: float, right: float, tolerance: float = 2e-8) -> bool:
    return abs(float(left) - float(right)) <= tolerance


def optimizer_steps(saved: dict[str, Any]) -> set[int]:
    values = set()
    for state in saved["optimizer"]["state"].values():
        step = state["step"]
        values.add(int(step.item()) if torch.is_tensor(step) else int(step))
    return values


def main() -> None:
    cfg = json.loads(CFG.read_text(encoding="utf-8"))
    report = ROOT / cfg["report_dir"]
    runs = ROOT / cfg["run_dir"]
    summary = json.loads((report / "checkpoint_metrics.json").read_text(encoding="utf-8"))
    with (report / "checkpoint_metrics.csv").open(newline="", encoding="utf-8") as stream:
        csv_rows = list(csv.DictReader(stream))
    rows = summary["metrics"]
    expected_keys = [(arm, epoch) for arm in ARMS for epoch in EPOCHS]
    assert summary["status"] == "complete" and len(rows) == len(csv_rows) == 24
    assert [(row["arm"], int(row["epoch"])) for row in rows] == expected_keys
    rows_by_key = {(row["arm"], int(row["epoch"])): row for row in rows}

    initialization = json.loads((report / "initialization.json").read_text(encoding="utf-8"))
    contract = json.loads((report / "contract.json").read_text(encoding="utf-8"))
    data_contract = json.loads((report / "data_contract.json").read_text(encoding="utf-8"))
    split = json.loads((report / "split.json").read_text(encoding="utf-8"))
    assert initialization["source"] == "random_initialization"
    assert initialization["trained_checkpoint_load_count"] == 0
    assert initialization["optimizer_state_entries_before_training"] == {arm: 0 for arm in ARMS}
    assert initialization["shared"]["all_arm_hash_maps_identical"] is True
    assert initialization["shared"]["seed"] == cfg["seed"] == 17
    assert initialization["shared"]["shared_learnable_combined_sha256"] == contract["shared_learnable_initial_sha256"]
    assert contract["initialization_source"] == "random_initialization"
    assert contract["forbidden_trained_checkpoint_loads"] == []
    assert data_contract["normalization_fit_trials"] == 2400
    assert data_contract["normalization_test_trials"] == 0
    assert len(split["train_trials"]) == len(split["train_images"]) == 2400
    assert len(split["test_trials"]) == len(split["test_images"]) == 1600
    assert split["image_intersection"] == []

    recomputed = []
    checkpoint_audit: list[dict[str, Any]] = []
    ordered_trial_ids: list[str] | None = None
    sampler_by_epoch: dict[int, set[str]] = {epoch: set() for epoch in range(1, 71)}
    dropout_by_epoch: dict[int, set[int]] = {epoch: set() for epoch in range(1, 71)}

    for arm in ARMS:
        arm_dir = runs / arm
        result = json.loads((arm_dir / "result.json").read_text(encoding="utf-8"))
        metrics = json.loads((arm_dir / "metrics.json").read_text(encoding="utf-8"))
        history = json.loads((arm_dir / "epochs.json").read_text(encoding="utf-8"))
        arm_init = json.loads((arm_dir / "initialization.json").read_text(encoding="utf-8"))
        first = json.loads((arm_dir / "first_update.json").read_text(encoding="utf-8"))
        assert result["contract"] == contract == summary["contract"]
        assert result["initialization_source"] == arm_init["source"] == "random_initialization"
        assert arm_init["trained_checkpoint_load_count"] == 0
        assert arm_init["shared_learnable_initial_sha256"] == contract["shared_learnable_initial_sha256"]
        assert result["optimizer"]["source"] == "new AdamW"
        assert result["optimizer"]["parameter_groups"] == 1
        assert result["global_steps"] == 2660 and result["checkpoint_count"] == 7
        assert result["scheduler"]["completed_steps"] == 2660
        assert close(result["scheduler"]["last_lr"], cfg["min_lr"], 1e-18)
        assert metrics == [rows_by_key[(arm, epoch)] for epoch in EPOCHS]
        assert len(history) == 70 and [item["epoch"] for item in history] == list(range(1, 71))
        required = {
            "projection", "electrode_embedding", "frequency_embedding", "transformer", "queries", "image_head"
        }
        if arm == "B0":
            required.add("coordinate_projection")
        if arm == "B2":
            required.add("fusion_P")
        assert set(first["gradient_norms_before_clip"]) == required
        assert first["global_step"] == 1 and close(first["lr"], cfg["peak_lr"] / cfg["warmup_steps"], 1e-18)
        assert all(float(value) > 0 for value in first["gradient_norms_before_clip"].values())
        assert all(first["before_sha256"][name] != first["after_sha256"][name] for name in required)
        assert all(result["initial_parameter_sha256"][name] != result["final_parameter_sha256"][name] for name in required)
        for item in history:
            assert item["coverage"] == 2400 and item["steps"] == 38
            assert item["global_step_start"] == (item["epoch"] - 1) * 38 + 1
            assert item["global_step_end"] == item["epoch"] * 38
            sampler_by_epoch[item["epoch"]].add(item["sampler_sha256"])
            dropout_by_epoch[item["epoch"]].add(item["dropout_seed"])

        for epoch in EPOCHS:
            prediction_path = arm_dir / "predictions" / f"epoch_{epoch:03d}.json"
            predictions = json.loads(prediction_path.read_text(encoding="utf-8"))
            assert len(predictions) == 1600
            trial_ids = [item["trial_id"] for item in predictions]
            assert len(set(trial_ids)) == 1600
            if ordered_trial_ids is None:
                ordered_trial_ids = trial_ids
            else:
                assert trial_ids == ordered_trial_ids
            metric = rows_by_key[(arm, epoch)]
            class_accuracy = sum(item["class_correct"] for item in predictions) / 1600
            image_top1 = sum(item["image_correct"] for item in predictions) / 1600
            latent_mse = sum(item["latent_mse"] for item in predictions) / 1600
            assert close(class_accuracy, metric["test_class_accuracy"], 1e-12)
            assert close(image_top1, metric["test_image_top1"], 1e-12)
            assert close(latent_mse, metric["test_latent_mse"])
            recomputed.append({
                "arm": arm, "epoch": epoch, "class_accuracy": class_accuracy,
                "image_top1": image_top1, "latent_mse": latent_mse,
            })

        checkpoints = sorted(arm_dir.glob("epoch_*.pt"))
        assert [int(path.stem.split("_")[-1]) for path in checkpoints] == list(CHECKPOINT_EPOCHS)
        for checkpoint_path, epoch in zip(checkpoints, CHECKPOINT_EPOCHS):
            saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            assert saved["task"] == "D080" and saved["initialization_source"] == "random_initialization"
            assert saved["arm"] == arm and saved["epoch"] == epoch and saved["seed"] == 17
            assert saved["contract"] == contract and saved["global_step"] == epoch * 38
            assert saved["scheduler"]["completed_steps"] == epoch * 38
            assert close(saved["scheduler"]["last_lr"], history[epoch - 1]["lr_last"], 1e-18)
            assert len(saved["history"]) == epoch and len(saved["metrics"]) == epoch // 10 + 1
            assert saved["normalization"]["fit_trials"] == 2400 and saved["normalization"]["test_trials"] == 0
            assert saved["split"] == split
            assert len(saved["optimizer"]["param_groups"]) == 1
            group = saved["optimizer"]["param_groups"][0]
            assert group["role"] == "all_trainable"
            assert close(group["lr"], saved["scheduler"]["last_lr"], 1e-18)
            steps = optimizer_steps(saved)
            assert steps == {epoch * 38}
            state_names = set(saved["model"])
            coordinate_names = {
                name for name in state_names
                if "coordinates" in name or ("position" in name and "frequency_position" not in name)
            }
            if arm == "B0":
                assert coordinate_names == {"encoder.coordinates", "encoder.position.weight", "encoder.position.bias"}
            else:
                assert not coordinate_names
            fusion_nonzero = None
            if arm == "B2":
                fusion = saved["model"]["encoder.fusion"]
                assert list(fusion.shape) == [16, 62]
                fusion_nonzero = bool((fusion.abs().sum(0) > 0).all())
                assert fusion_nonzero
            else:
                assert "encoder.fusion" not in state_names
            assert model_hash(saved["model"]) == rows_by_key[(arm, epoch)]["model_state_sha256"]
            checkpoint_audit.append({
                "arm": arm, "epoch": epoch, "optimizer_steps": sorted(steps),
                "coordinate_state_keys": sorted(coordinate_names),
                "fusion_all_62_columns_nonzero": fusion_nonzero,
            })

    assert all(len(value) == 1 for value in sampler_by_epoch.values())
    assert all(len(value) == 1 for value in dropout_by_epoch.values())
    audit = {
        "status": "PASS",
        "arms": 3,
        "checkpoints": 21,
        "evaluations": 24,
        "predictions_per_evaluation": 1600,
        "metrics_recomputed_from_predictions": True,
        "common_ordered_test_trials": True,
        "train_only_normalization_2400_test_0": True,
        "no_train_test_image_overlap": True,
        "random_initialization_no_trained_checkpoint_load": True,
        "shared_random_learnable_state_identical": True,
        "same_sampler_hash_each_epoch_across_arms": True,
        "same_dropout_seed_each_epoch_across_arms": True,
        "one_new_adamw_group_all_trainable": True,
        "optimizer_steps_match_38_steps_per_epoch": True,
        "warmup_and_cosine_endpoints_match": True,
        "B1_B2_coordinate_state_absent": True,
        "B2_all_62_fusion_columns_nonzero_at_every_checkpoint": True,
        "required_first_and_endpoint_parameters_updated": True,
        "recomputed_metrics": recomputed,
        "checkpoint_audit": checkpoint_audit,
    }
    (report / "AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in audit.items() if key not in {"recomputed_metrics", "checkpoint_audit"}}, indent=2))


if __name__ == "__main__":
    main()
