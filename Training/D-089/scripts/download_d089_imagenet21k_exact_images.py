"""Download exact D089 ImageNet-21K stimuli using parquet ID statistics.

Only row groups whose `id` min/max can contain a requested image are fetched.
Every extracted payload is checked against the frozen D057 source manifest.
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import io
import json
import os
import threading
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from download_d089_bp_exact_images import RemoteRanges, atomic_json


ROOT = Path(__file__).resolve().parents[1]
TICKET_PATH = ROOT / ".runtime/d056_download_ticket.json"
SOURCE_MANIFEST = ROOT / "reports/d057_stimulus_sources.json"
OUTPUT_ROOT = ROOT / "data/stimulus-images"
PROGRESS_ROOT = ROOT / ".runtime/d089_imagenet21k_download_completed"
STATUS_PATH = ROOT / "reports/d089_imagenet21k_image_download_status.json"
LOCK_DIR = ROOT / ".runtime/d089_imagenet21k_image_download.lock"


def main() -> None:
    try:
        LOCK_DIR.mkdir(parents=True)
    except FileExistsError:
        raise SystemExit("D089 ImageNet-21K downloader is already running")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    PROGRESS_ROOT.mkdir(parents=True, exist_ok=True)
    ticket = json.loads(TICKET_PATH.read_text(encoding="utf-8"))
    manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    expected = {
        item["image_id"]: item
        for item in manifest["images"]
        if item["repository"] == ticket["repository"]
    }
    expected_stems = {Path(image_id).stem: image_id for image_id in expected}
    if len(expected) != 3504:
        raise RuntimeError(f"expected 3504 ImageNet-21K records, found {len(expected)}")

    print_lock = threading.Lock()
    write_lock = threading.Lock()

    def emit(**value: object) -> None:
        with print_lock:
            print(json.dumps(value, sort_keys=True), flush=True)

    def valid(image_id: str) -> bool:
        path = OUTPUT_ROOT / image_id
        contract = expected[image_id]
        if not path.is_file() or path.stat().st_size != int(contract["bytes"]):
            return False
        return hashlib.sha256(path.read_bytes()).hexdigest() == contract["sha256"]

    shard_targets: dict[int, set[str]] = {int(job["index"]): set() for job in ticket["shards"]}
    for image_id in expected:
        class_id = image_id.split("_", 1)[0]
        for shard_index in ticket["class_to_shards"][class_id]:
            shard_targets[int(shard_index)].add(Path(image_id).stem)

    def work(job: dict) -> dict:
        shard_index = int(job["index"])
        record_path = PROGRESS_ROOT / (job["filename"] + ".json")
        candidates = {
            stem for stem in shard_targets[shard_index]
            if not valid(expected_stems[stem])
        }
        if not candidates:
            record = {"file": job["filename"], "status": "already_valid", "images": 0}
            atomic_json(record_path, record)
            return record

        url = ticket["source_base"] + job["filename"]
        remote = RemoteRanges(url, int(job["bytes"]))
        metadata = pq.read_metadata(pa.PythonFile(remote))
        groups: dict[int, set[str]] = {}
        id_column = None
        for column_index in range(len(metadata.schema.to_arrow_schema())):
            if metadata.schema.column(column_index).path == "id":
                id_column = column_index
                break
        if id_column is None:
            raise RuntimeError(f"id column absent from {job['filename']}")
        for group_index in range(metadata.num_row_groups):
            stats = metadata.row_group(group_index).column(id_column).statistics
            if stats is None or not stats.has_min_max:
                groups[group_index] = set(candidates)
                continue
            minimum = stats.min.decode() if isinstance(stats.min, bytes) else str(stats.min)
            maximum = stats.max.decode() if isinstance(stats.max, bytes) else str(stats.max)
            selected = {stem for stem in candidates if minimum <= stem <= maximum}
            if selected:
                groups[group_index] = selected

        parquet = pq.ParquetFile(pa.PythonFile(remote), metadata=metadata, pre_buffer=False, buffer_size=0)
        found: list[dict] = []
        try:
            for group_index, wanted in groups.items():
                table = parquet.read_row_group(group_index, columns=["id", "image"])
                ids = table.column("id").to_pylist()
                images = table.column("image")
                for row_index, stem in enumerate(ids):
                    if stem not in wanted:
                        continue
                    image_id = expected_stems[stem]
                    data = images[row_index].as_py()
                    if isinstance(data, dict):
                        data = data["bytes"]
                    contract = expected[image_id]
                    digest = hashlib.sha256(data).hexdigest()
                    if len(data) != int(contract["bytes"]) or digest != contract["sha256"]:
                        raise RuntimeError(f"content contract failed for {image_id}")
                    with Image.open(io.BytesIO(data)) as image:
                        image.load()
                        geometry = list(image.size)
                        image_format = image.format
                    with write_lock:
                        if not valid(image_id):
                            final = OUTPUT_ROOT / image_id
                            partial = final.with_suffix(final.suffix + ".d089partial")
                            partial.write_bytes(data)
                            os.replace(partial, final)
                    found.append(
                        {
                            "image_id": image_id,
                            "bytes": len(data),
                            "sha256": digest,
                            "size": geometry,
                            "format": image_format,
                            "row_group": group_index,
                            "row": row_index,
                        }
                    )
                del table
        finally:
            parquet.close()
            remote.close()

        record = {
            "file": job["filename"],
            "status": "complete",
            "repository": ticket["repository"],
            "revision": ticket["revision"],
            "candidate_ids": len(candidates),
            "selected_row_groups": len(groups),
            "images": found,
        }
        atomic_json(record_path, record)
        emit(
            event="row_groups_complete",
            file=job["filename"],
            groups=len(groups),
            images=len(found),
        )
        return record

    try:
        jobs = [job for job in ticket["shards"] if shard_targets[int(job["index"])]]
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            outcomes = list(pool.map(work, jobs))
        invalid = [image_id for image_id in sorted(expected) if not valid(image_id)]
        status = {
            "status": "complete" if not invalid else "partial",
            "repository": ticket["repository"],
            "revision": ticket["revision"],
            "expected": len(expected),
            "valid": len(expected) - len(invalid),
            "invalid_or_missing": invalid,
            "jobs": len(outcomes),
        }
        atomic_json(STATUS_PATH, status)
        emit(event="source_complete", **{key: value for key, value in status.items() if key != "invalid_or_missing"})
        if invalid:
            raise RuntimeError(f"{len(invalid)} exact images remain invalid or missing")
    finally:
        try:
            LOCK_DIR.rmdir()
        except OSError:
            pass


if __name__ == "__main__":
    main()
