"""Fast local static and functional preflight for D078's new files."""

from __future__ import annotations

import ast
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from d078_augmentation import self_test as augmentation_self_test


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    cfg_path = ROOT / "config/d078_a100_continuation.json"
    runner_path = ROOT / "scripts/d078_run_continuation.py"
    identity_path = ROOT / "scripts/d078_fit_identity.py"
    aug_path = ROOT / "src/d078_augmentation.py"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    if cfg["arms"] != ["A0", "A1", "A2", "A3"] or cfg["checkpoint_epochs"] != [80, 90, 100, 110, 120]:
        raise AssertionError("arm/checkpoint contract differs")
    if cfg["augmentation"]["A1"]["global_sigma"] != 0.05 or cfg["augmentation"]["A2"]["sensitive_total_sigma"] != 0.15:
        raise AssertionError("noise contract differs")
    if cfg["augmentation"]["A3"]["mask_sensitive"] != 124:
        raise AssertionError("A3 mask count differs")
    runner = runner_path.read_text(encoding="utf-8")
    tree = ast.parse(runner)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "apply_augmentation"]
    if len(calls) != 1:
        raise AssertionError("runner must have exactly one augmentation call in its train loop")
    if "model(x[batch_indices])" not in runner or "model(augmented)" not in runner:
        raise AssertionError("clean evaluation or augmented training call missing")
    if "optimizer.load_state_dict(parent[\"optimizer\"])" not in runner:
        raise AssertionError("parent optimizer is not loaded")
    functional = augmentation_self_test()
    output = {
        "status": "PASS", "functional": functional,
        "files": {str(path.relative_to(ROOT)): sha(path) for path in (cfg_path, runner_path, identity_path, aug_path,
                  ROOT / "scripts/d078_locked.py", ROOT / "scripts/d078_remote_supervisor.sh")},
        "static": {"augmentation_train_call_count": 1, "clean_eval_call_present": True,
                   "parent_optimizer_load_present": True, "checkpoints": cfg["checkpoint_epochs"]},
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
