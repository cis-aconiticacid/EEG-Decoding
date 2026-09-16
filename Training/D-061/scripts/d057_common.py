"""Small shared IO/contracts helpers. No model construction or CUDA at import."""
import csv
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]
CONFIG = ROOT / "config/d057_raev2_latent_s17.json"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def emit(event, **fields):
    print(json.dumps({"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                      "event": event, **fields}, ensure_ascii=False, allow_nan=False), flush=True)


def config():
    return json.loads(CONFIG.read_text())


def status(cfg, state, **fields):
    value = {"task_id": "D057", "status": state, "updated_unix": time.time(),
             "final_holdout_evaluated": False, **fields}
    atomic_json(ROOT / cfg["status"], value)
    emit(state, **fields)


def data_contract(cfg):
    source = ROOT / cfg["source_config"]
    if sha(source) != cfg["source_config_sha256"]:
        raise ValueError("D055 split configuration changed")
    src = json.loads(source.read_text())
    for key in ("image_assignments", "trial_split", "channel_map"):
        if sha(ROOT / src[key]) != src[key + "_sha256"]:
            raise ValueError(key + " hash mismatch")
    with (ROOT / src["image_assignments"]).open(newline="", encoding="utf-8") as f:
        images = list(csv.DictReader(f))
    with (ROOT / src["trial_split"]).open(newline="", encoding="utf-8") as f:
        trials = [r for r in csv.DictReader(f) if r["d055_split"] in ("train", "monitor_val")]
    trials.sort(key=lambda r: (r["d055_split"] != "train", int(r["cache_index"])))
    if len(images) != 4000 or len({r['image_id'] for r in images}) != 4000 or len(trials) != 60656:
        raise ValueError("full-data counts differ")
    for split, ni, nt in (("train", 3600, 57465), ("monitor_val", 200, 3191)):
        if sum(r["d055_split"] == split for r in images) != ni:
            raise ValueError("image split count changed")
        rows = [r for r in trials if r["d055_split"] == split]
        if len(rows) != nt or len({r['subject'] for r in rows}) != 16:
            raise ValueError("trial split count/subjects changed")
    return src, images, trials


def require_gpu0(cfg):
    # The existing lock selects the physical GPU by UUID, exposed as logical cuda:0.
    expected = cfg.get("gpu_uuid")
    if not expected:
        return
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != expected:
        raise RuntimeError("D057 must execute under shared UUID lock")
    row = subprocess.check_output(["nvidia-smi", "-i", "0", "--query-gpu=uuid,name",
                                   "--format=csv,noheader"], text=True).strip()
    if not row.startswith(expected + ",") or "A100" not in row:
        raise RuntimeError("physical GPU0 identity differs from approved A100")


def code_contract(paths):
    resolved = {}
    for path in map(Path, paths):
        base = PROJECT_ROOT if path.parts[:2] == ("src", "eegdecoding") else ROOT
        resolved[str(path)] = sha(base / path)
    return resolved
