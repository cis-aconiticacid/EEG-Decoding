"""Download and SHA-256 verify the official EEG-ImageNet v1 archives."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eegdecoding.data import ASSETS, ensure_archive, verify_archive  # noqa: E402


PARTS = {
    "1": ("EEG-ImageNet_1.pth",),
    "2": ("EEG-ImageNet_2.pth",),
    "all": ("EEG-ImageNet_1.pth", "EEG-ImageNet_2.pth"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download official EEG-ImageNet archives into the repository data directory."
    )
    parser.add_argument(
        "--part",
        choices=sorted(PARTS),
        default="1",
        help="archive part to download; D-096 requires part 1 (default: 1)",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="verify existing files and fail instead of downloading missing files",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = []
    for name in PARTS[args.part]:
        path = verify_archive(name) if args.verify_only else ensure_archive(name)
        records.append(
            {
                "name": name,
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": ASSETS[name]["sha256"],
                "status": "verified",
            }
        )
    print(json.dumps(records, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
