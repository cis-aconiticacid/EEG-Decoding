"""Audit D079 checkpoints, predictions, fixed evaluation rows, and fairness invariants."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "config/d079_no_coord_early_fusion.json"
ARMS = ("B0", "B1", "B2")
EPOCHS = (70, 80, 90, 100, 110, 120)
CHECKPOINT_EPOCHS = EPOCHS[1:]


def model_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def close(left: float, right: float, tolerance: float = 2e-8) -> bool:
    return abs(left - right) <= tolerance


def main() -> None:
    cfg = json.loads(CFG.read_text(encoding="utf-8"))
    report = ROOT / cfg["report_dir"]
    runs = ROOT / cfg["run_dir"]
    summary = json.loads((report / "checkpoint_metrics.json").read_text(encoding="utf-8"))
    with (report / "checkpoint_metrics.csv").open(newline="", encoding="utf-8") as stream:
        csv_rows = list(csv.DictReader(stream))
    rows = summary["metrics"]
    assert summary["status"] == "complete" and len(rows) == len(csv_rows) == 18
    assert [(row["arm"], int(row["epoch"])) for row in rows] == [
        (arm, epoch) for arm in ARMS for epoch in EPOCHS
    ]

    ordered_trial_ids: list[str] | None = None
    recomputed = []
    rows_by_key = {(row["arm"], int(row["epoch"])): row for row in rows}
    sampler_by_epoch: dict[int, set[str]] = {epoch: set() for epoch in range(71, 121)}
    dropout_seed_by_epoch: dict[int, set[int]] = {epoch: set() for epoch in range(71, 121)}
    checkpoint_audit: list[dict[str, Any]] = []

    for arm in ARMS:
        arm_dir = runs / arm
        result = json.loads((arm_dir / "result.json").read_text(encoding="utf-8"))
        metrics = json.loads((arm_dir / "metrics.json").read_text(encoding="utf-8"))
        history = json.loads((arm_dir / "epochs.json").read_text(encoding="utf-8"))
        initialization = json.loads((arm_dir / "initialization.json").read_text(encoding="utf-8"))
        first_update = json.loads((arm_dir / "first_update.json").read_text(encoding="utf-8"))
        assert result["contract"] == summary["contract"] and result["augmentation"] == "none"
        assert metrics == [rows_by_key[(arm, epoch)] for epoch in EPOCHS]
        assert len(history) == 50 and [row["epoch"] for row in history] == list(range(71, 121))
        assert result["optimizer"]["source"] == "new AdamW; no inherited state"
        assert initialization["optimizer"] == "new AdamW with empty state"
        required = {"projection", "transformer", "image_head"} | ({"fusion_P"} if arm == "B2" else set())
        assert set(first_update["gradient_norms_before_clip"]) == required
        assert all(value > 0 for value in first_update["gradient_norms_before_clip"].values())
        assert all(first_update["before_sha256"][name] != first_update["after_sha256"][name] for name in required)
        assert all(result["initial_parameter_sha256"][name] != result["final_parameter_sha256"][name] for name in required)
        for item in history:
            sampler_by_epoch[item["epoch"]].add(item["sampler_sha256"])
            dropout_seed_by_epoch[item["epoch"]].add(item["dropout_seed"])

        for epoch in EPOCHS:
            prediction_path = arm_dir / "predictions" / f"epoch_{epoch:03d}.json"
            predictions = json.loads(prediction_path.read_text(encoding="utf-8"))
            assert len(predictions) == 1600
            trial_ids = [row["trial_id"] for row in predictions]
            assert len(set(trial_ids)) == 1600
            if ordered_trial_ids is None:
                ordered_trial_ids = trial_ids
            else:
                assert trial_ids == ordered_trial_ids
            metric = rows_by_key[(arm, epoch)]
            class_accuracy = sum(row["class_correct"] for row in predictions) / 1600
            image_top1 = sum(row["image_correct"] for row in predictions) / 1600
            latent_mse = sum(row["latent_mse"] for row in predictions) / 1600
            assert close(class_accuracy, metric["test_class_accuracy"], 1e-12)
            assert close(image_top1, metric["test_image_top1"], 1e-12)
            assert close(latent_mse, metric["test_latent_mse"])
            recomputed.append({
                "arm": arm,
                "epoch": epoch,
                "class_accuracy": class_accuracy,
                "image_top1": image_top1,
                "latent_mse": latent_mse,
            })

        checkpoints = sorted(arm_dir.glob("epoch_*.pt"))
        assert len(checkpoints) == 5
        for checkpoint_path, epoch in zip(checkpoints, CHECKPOINT_EPOCHS):
            saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            assert saved["arm"] == arm and saved["epoch"] == epoch and saved["seed"] == 17
            assert saved["contract"] == summary["contract"]
            assert saved["cursor"]["completed_steps"] == (epoch - 70) * 38
            assert len(saved["history"]) == epoch - 70
            state_names = set(saved["model"])
            coordinate_names = {
                name for name in state_names
                if "coordinates" in name or ("position" in name and "frequency_position" not in name)
            }
            if arm == "B0":
                assert coordinate_names == {"encoder.coordinates", "encoder.position.weight", "encoder.position.bias"}
            else:
                assert not coordinate_names
            if arm == "B2":
                fusion = saved["model"]["encoder.fusion"]
                assert list(fusion.shape) == [16, 62]
                assert bool((fusion.abs().sum(0) > 0).all())
            else:
                assert "encoder.fusion" not in state_names
            roles = [group["role"] for group in saved["optimizer"]["param_groups"]]
            expected_roles = ["existing_encoder", "image_head"] + (["fusion_P"] if arm == "B2" else [])
            assert roles == expected_roles
            expected_lrs = [cfg["encoder_lr"], cfg["head_lr"]] + ([cfg["fusion_lr"]] if arm == "B2" else [])
            assert [group["lr"] for group in saved["optimizer"]["param_groups"]] == expected_lrs
            steps = {int(value["step"].item()) for value in saved["optimizer"]["state"].values()}
            assert steps == {(epoch - 70) * 38}
            assert model_hash(saved["model"]) == rows_by_key[(arm, epoch)]["model_state_sha256"]
            checkpoint_audit.append({
                "arm": arm,
                "epoch": epoch,
                "optimizer_steps": sorted(steps),
                "coordinate_state_keys": sorted(coordinate_names),
                "fusion_all_62_columns_nonzero": (
                    bool((saved["model"]["encoder.fusion"].abs().sum(0) > 0).all()) if arm == "B2" else None
                ),
            })

    assert all(len(values) == 1 for values in sampler_by_epoch.values())
    assert all(len(values) == 1 for values in dropout_seed_by_epoch.values())
    assert rows_by_key[("B0", 70)]["test_class_accuracy"] == 0.37125
    assert rows_by_key[("B0", 70)]["test_image_top1"] == 0.018125
    assert close(rows_by_key[("B0", 70)]["test_latent_mse"], 0.9566891086101532, 1e-12)

    audit = {
        "status": "PASS",
        "arms": 3,
        "checkpoints": 15,
        "evaluations": 18,
        "predictions_per_evaluation": 1600,
        "metrics_recomputed_from_predictions": True,
        "common_ordered_test_trials": True,
        "same_sampler_hash_each_epoch_across_arms": True,
        "same_dropout_seed_each_epoch_across_arms": True,
        "new_optimizer_steps_match_38_steps_per_epoch": True,
        "B0_parent_reproduction": True,
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
