"""Download the exact Benjamin Paine ImageNet-1K stimulus rows for D089.

This Windows-local downloader reads only the required parquet row groups, checks
every extracted image against the frozen D057 source manifest, and writes via an
atomic partial file.  It uses CPU/network resources and can run beside EEG GPU
pretraining.
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import io
import json
import os
import threading
import time
import urllib.request
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SOURCE_INDEX = ROOT / "reports/d057_image_supplement_index.json"
SOURCE_MANIFEST = ROOT / "reports/d057_stimulus_sources.json"
FOOTER_ROOT = ROOT / ".runtime/d057/image_sources/bp_metadata"
OUTPUT_ROOT = ROOT / "data/stimulus-images"
PROGRESS_ROOT = ROOT / ".runtime/d089_bp_download_completed"
STATUS_PATH = ROOT / "reports/d089_bp_image_download_status.json"
LOCK_DIR = ROOT / ".runtime/d089_bp_image_download.lock"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(partial, path)


class RemoteRanges(io.RawIOBase):
    def __init__(self, url: str, size: int):
        self.url = url
        self.size = int(size)
        self.pos = 0
        self.opener = urllib.request.build_opener()

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            self.pos = offset
        elif whence == 1:
            self.pos += offset
        elif whence == 2:
            self.pos = self.size + offset
        else:
            raise ValueError(whence)
        return self.pos

    def read(self, n: int = -1) -> bytes:
        end = self.size if n < 0 else min(self.pos + n, self.size)
        if end <= self.pos:
            return b""
        start = self.pos
        request = urllib.request.Request(
            self.url,
            headers={"Range": f"bytes={start}-{end - 1}", "User-Agent": "Mozilla/5.0"},
        )
        for attempt in range(5):
            try:
                with self.opener.open(request, timeout=180) as response:
                    data = response.read()
                    content_range = response.headers.get("Content-Range")
                    if response.status != 206:
                        raise RuntimeError(f"range ignored: HTTP {response.status}")
                    expected_range = f"bytes {start}-{end - 1}/{self.size}"
                    if content_range != expected_range:
                        raise RuntimeError(f"Content-Range {content_range!r} != {expected_range!r}")
                if len(data) != end - start:
                    raise RuntimeError(f"short range: {len(data)} != {end - start}")
                self.pos = end
                return data
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(2**attempt)
        raise AssertionError("unreachable")


def main() -> None:
    try:
        LOCK_DIR.mkdir(parents=True)
    except FileExistsError:
        raise SystemExit("D089 Benjamin-Paine image downloader is already running")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    PROGRESS_ROOT.mkdir(parents=True, exist_ok=True)
    source = json.loads(SOURCE_INDEX.read_text(encoding="utf-8"))
    manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    expected = {
        item["image_id"]: item
        for item in manifest["images"]
        if item["repository"] == source["repository"]
    }
    requested = {match["image_id"] for job in source["jobs"] for match in job["matches"]}
    if requested != set(expected):
        raise RuntimeError(
            f"source contract mismatch: row index={len(requested)}, manifest={len(expected)}"
        )

    output_lock = threading.Lock()

    def emit(**value: object) -> None:
        with output_lock:
            print(json.dumps(value, sort_keys=True), flush=True)

    def existing_is_valid(image_id: str) -> bool:
        path = OUTPUT_ROOT / image_id
        if not path.is_file() or path.stat().st_size != int(expected[image_id]["bytes"]):
            return False
        return hashlib.sha256(path.read_bytes()).hexdigest() == expected[image_id]["sha256"]

    def work(job: dict) -> dict:
        record_path = PROGRESS_ROOT / (job["filename"] + ".json")
        pending = [match for match in job["matches"] if not existing_is_valid(match["image_id"])]
        if not pending:
            record = {"file": job["filename"], "status": "already_valid", "images": len(job["matches"])}
            atomic_json(record_path, record)
            return record

        footer_path = FOOTER_ROOT / (job["filename"] + ".footer")
        metadata = pq.read_metadata(pa.BufferReader(footer_path.read_bytes()))
        remote = RemoteRanges(source["base"] + job["filename"], int(job["bytes"]))
        parquet = pq.ParquetFile(pa.PythonFile(remote), metadata=metadata, pre_buffer=False, buffer_size=0)
        saved = []
        try:
            for group in sorted({int(match["group"]) for match in pending}):
                matches = [match for match in pending if int(match["group"]) == group]
                table = parquet.read_row_group(group, columns=["image"])
                column = table.column("image")
                for match in matches:
                    entry = column[int(match["row"])].as_py()
                    if entry["path"] != match["stored_path"]:
                        raise RuntimeError(
                            f"row mismatch for {match['image_id']}: {entry['path']} != {match['stored_path']}"
                        )
                    data = entry["bytes"]
                    contract = expected[match["image_id"]]
                    digest = sha256_bytes(data)
                    if len(data) != int(contract["bytes"]) or digest != contract["sha256"]:
                        raise RuntimeError(f"content contract failed for {match['image_id']}")
                    with Image.open(io.BytesIO(data)) as image:
                        image.load()
                        geometry = list(image.size)
                        image_format = image.format
                    final = OUTPUT_ROOT / match["image_id"]
                    partial = final.with_suffix(final.suffix + ".d089partial")
                    partial.write_bytes(data)
                    os.replace(partial, final)
                    saved.append(
                        {
                            "image_id": match["image_id"],
                            "bytes": len(data),
                            "sha256": digest,
                            "size": geometry,
                            "format": image_format,
                        }
                    )
        finally:
            parquet.close()
            remote.close()

        record = {
            "file": job["filename"],
            "status": "complete",
            "repository": source["repository"],
            "revision": source["revision"],
            "images": saved,
        }
        atomic_json(record_path, record)
        emit(event="row_groups_complete", file=job["filename"], images=len(saved))
        return record

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            outcomes = list(pool.map(work, source["jobs"]))
        invalid = [image_id for image_id in sorted(expected) if not existing_is_valid(image_id)]
        status = {
            "status": "complete" if not invalid else "partial",
            "repository": source["repository"],
            "revision": source["revision"],
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
