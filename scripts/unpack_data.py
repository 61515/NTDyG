#!/usr/bin/env python3
"""Unpack the included processed datasets and verify their original CSV hashes."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def unpack(record: dict, output_root: Path) -> None:
    archive = ROOT / record["archive"]
    destination = output_root / record["csv"]
    if destination.exists():
        if sha256(destination) != record["csv_sha256"]:
            raise ValueError(f"Existing file has a different checksum: {destination}")
        print(f"Verified existing {record['csv']}")
        return
    if sha256(archive) != record["archive_sha256"]:
        raise ValueError(f"Archive checksum mismatch: {archive}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    size = 0
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, prefix=destination.name + ".", suffix=".tmp", delete=False
        ) as output:
            temporary = Path(output.name)
            with gzip.open(archive, "rb") as source:
                while block := source.read(1024 * 1024):
                    output.write(block)
                    digest.update(block)
                    size += len(block)
        if size != record["csv_bytes"] or digest.hexdigest() != record["csv_sha256"]:
            raise ValueError(f"Unpacked CSV checksum or size mismatch: {archive}")
        os.replace(temporary, destination)
        print(f"Unpacked and verified {record['csv']} ({size:,} bytes)")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema", choices=("all", "binary", "multiclass"), default="all")
    parser.add_argument("--dataset", choices=("all", "IIoTAPT", "Darknet"), default="all")
    parser.add_argument("--output-root", type=Path, default=ROOT,
                        help="Release root; CSVs are written below its data/ directory.")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "configs/processed_data_archives.json").read_text())
    for record in manifest["files"]:
        if args.schema != "all" and record["schema"] != args.schema:
            continue
        if args.dataset != "all" and record["dataset"] != args.dataset:
            continue
        unpack(record, args.output_root)


if __name__ == "__main__":
    main()
