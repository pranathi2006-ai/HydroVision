#!/usr/bin/env python3
"""Register Phase 3 public/synthetic images and their provenance in SQLite."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.store import Store  # noqa: E402


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Import a licensed Phase 3 image tree without copying or uploading pixels."
    )
    parser.add_argument("dataset_key")
    parser.add_argument("image_root", type=Path)
    parser.add_argument("--database", type=Path, default=ROOT / "data" / "hydrovision.sqlite3")
    parser.add_argument("--registry", type=Path, default=ROOT / "training" / "phase3_dataset_registry.json")
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()

    registry = {item["dataset_key"]: item for item in json.loads(args.registry.read_text())}
    if args.dataset_key not in registry:
        raise SystemExit(f"Unknown dataset_key {args.dataset_key!r}; add a licensed registry entry first")
    if not args.image_root.is_dir():
        raise SystemExit(f"Image root does not exist: {args.image_root}")

    store = Store(args.database)
    dataset_id = store.upsert_training_dataset(registry[args.dataset_key])
    imported = duplicates = 0
    for path in sorted(args.image_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        split = next((part for part in path.parts if part in {"train", "val", "test"}), args.split)
        created = store.insert_training_image(dataset_id, {
            "source_ref": str(path.relative_to(args.image_root)),
            "content_hash": digest,
            "split": split,
            "synthetic": args.synthetic,
        })
        imported += int(created)
        duplicates += int(not created)
    print(f"Registered {imported} images ({duplicates} duplicates skipped) for {args.dataset_key}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
