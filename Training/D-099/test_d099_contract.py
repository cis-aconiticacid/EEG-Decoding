import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
D099 = json.loads((ROOT / "Training/D-099/config/config.json").read_text(encoding="utf-8"))
D098 = json.loads((ROOT / "Training/D-098/config/config.json").read_text(encoding="utf-8"))


def test_participant_12_full_frequency_contract():
    assert D099["subject"] == 12
    assert D099["archive"] == "EEG-ImageNet_2.pth"
    assert D099["frequency_mask_below_hz"] is None
    assert D099["add_position_after_cross_attention"] is True


def test_only_intended_training_controls_change():
    shared = (
        "seed",
        "crop",
        "classes",
        "cross_attention",
        "block_dropout",
        "head_dropout",
        "logical_batch",
        "microbatch_candidates",
        "peak_memory_limit_gib",
        "learning_rate",
        "minimum_learning_rate",
        "warmup_epochs",
        "weight_decay",
        "label_smoothing",
        "gradient_clip",
        "epochs",
        "checkpoint_every_epochs",
        "augmentation",
        "precision",
        "objective",
    )
    for key in shared:
        assert D099[key] == D098[key]
