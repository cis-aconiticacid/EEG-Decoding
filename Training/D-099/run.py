"""Train the participant-12 full-frequency control for D-099."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import torch


EXPERIMENT_DIR = Path(__file__).resolve().parent
ROOT = EXPERIMENT_DIR.parents[1]
D098_DIR = EXPERIMENT_DIR.parent / "D-098"
CONFIG = EXPERIMENT_DIR / "config/config.json"

sys.path.insert(0, str(D098_DIR))
SPEC = importlib.util.spec_from_file_location("d098_training_run", D098_DIR / "run.py")
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"cannot load D-098 training implementation from {D098_DIR}")
d098 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = d098
SPEC.loader.exec_module(d098)
# D-098 is symlinked into the isolated D-099 remote workspace.  Keep its
# reusable path serializer anchored to the active D-099 checkout.
d098.ROOT = ROOT

Config = dict[str, Any]


def load_coordinates(cfg: Config) -> tuple[torch.Tensor, dict[str, Any]]:
    """Reuse the D-098 montage while recording its stable repository path."""
    path = D098_DIR / cfg["channel_map"]
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = sorted(csv.DictReader(stream), key=lambda row: int(row["tensor_index"]))
    if [int(row["tensor_index"]) for row in rows] != list(range(d098.CHANNELS)):
        raise AssertionError("channel_map tensor order is not exactly 0..61")
    names = [str(row["canonical_name"]) for row in rows]
    columns = cfg["coordinate_columns"]
    coordinates = torch.tensor(
        [[float(row[column]) for column in columns] for row in rows], dtype=torch.float32
    )
    normalized = d098.normalize_unit_sphere_coordinates(coordinates)
    return normalized, {
        "path": "Training/D-098/config/channel_map.csv",
        "file_sha256": d098.sha256(path),
        "columns": columns,
        "axes": cfg["coordinate_axes"],
        "source": "MNE colin27_1005 template-fit approximation in D-098 channel_map",
        "channel_names": names,
        "raw_coordinate_sha256": d098.tensor_sha256(coordinates),
        "normalized_coordinate_sha256": d098.tensor_sha256(normalized),
        "normalization": "row-wise r / ||r||_2; no centering and no axis change",
        "normalized_radius_min": float(normalized.norm(dim=-1).min()),
        "normalized_radius_max": float(normalized.norm(dim=-1).max()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--participant", type=int, default=12, choices=[12])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--run", action="store_true")
    return parser.parse_args()


def load_config(path: Path, participant: int) -> Config:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if participant != 12 or cfg["subject"] != 12:
        raise AssertionError("D099 is the participant-12 control")
    if cfg["crop"] != [40, 440] or cfg["classes"] != 80:
        raise AssertionError("D099 crop/class contract differs")
    if cfg["frequency_mask_below_hz"] is not None:
        raise AssertionError("D099 must retain the complete 12--80 Hz frequency branch")
    if cfg["add_position_after_cross_attention"] is not True:
        raise AssertionError("D099 retains post-cross-attention position")
    if cfg["label_smoothing"] != 0.0:
        raise AssertionError("D099 requires ordinary cross entropy")
    return cfg


def prepare_output_dirs(cfg: Config) -> tuple[Path, Path]:
    run_dir = EXPERIMENT_DIR / cfg["run_dir"]
    report_dir = EXPERIMENT_DIR / cfg["report_dir"]
    run_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, report_dir


def build_contract(
    cfg: Config,
    config_path: Path,
    coordinate_record: dict[str, Any],
    split: dict[str, Any],
) -> dict[str, Any]:
    return {
        "excluded_band_hz": cfg.get("excluded_band_hz"),
        "exclusion_protocol": cfg.get("exclusion_protocol"),
        "post_fusion_projection": cfg.get("post_fusion_projection", False),
        "frequency_mask_below_hz": cfg.get("frequency_mask_below_hz"),
        "add_position_after_cross_attention": cfg["add_position_after_cross_attention"],
        "participant": 12,
        "archive": cfg["archive"],
        "present_classes": split["present_classes"],
        "missing_classes": split["missing_classes"],
        "task": cfg["task"],
        "config_path": d098.workspace_relative(config_path),
        "config_sha256": d098.sha256(config_path),
        "model_path": "Training/D-098/model.py",
        "model_sha256": d098.sha256(D098_DIR / "model.py"),
        "runner_sha256": d098.sha256(Path(__file__)),
        "montage_sha256": coordinate_record["file_sha256"],
        "official_train_hash": split["official_train_hash"],
        "test_hash": split["test_hash"],
        "layout": "Training/D-099",
    }


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    cfg = load_config(config_path, args.participant)
    device = d098.configure_cuda(cfg)
    run_dir, report_dir = prepare_output_dirs(cfg)

    coordinates, coordinate_record = load_coordinates(cfg)
    microbatch_record = d098.probe_microbatch(coordinates, cfg, device)
    d098.set_seed(cfg["seed"])
    model = d098.build_model(coordinates, cfg, device)
    prepared = d098.prepare_data(cfg, model)
    split = d098.validate_split(prepared)
    contract = build_contract(cfg, config_path, coordinate_record, split)
    preflight = d098.build_preflight(
        cfg, model, prepared, coordinate_record, split, contract, microbatch_record
    )
    d098.atomic_json(report_dir / "preflight.json", preflight)

    microbatch = int(microbatch_record["microbatch"])
    if args.smoke:
        return d098.run_smoke(model, prepared, cfg, device, microbatch, report_dir)

    history, started = d098.train_model(
        model, prepared, cfg, device, microbatch, run_dir, report_dir, contract
    )
    test_metrics, predictions = d098.evaluate_test_once(
        model,
        prepared["test"],
        prepared["test_labels"],
        prepared["test_indices"],
        prepared["data"],
        cfg,
        device,
        microbatch,
    )
    prediction_path = run_dir / "test_predictions.jsonl"
    d098.write_predictions(prediction_path, predictions)
    result = d098.build_result(
        model,
        cfg,
        history,
        test_metrics,
        started,
        microbatch,
        run_dir,
        prediction_path,
        contract,
    )
    d098.atomic_json(run_dir / "result.json", result)
    d098.atomic_json(report_dir / "status.json", result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
