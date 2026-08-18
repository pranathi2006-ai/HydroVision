#!/usr/bin/env python3
"""Merge source YOLO datasets into HydroVision's two-class schema."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import yaml


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
TARGET_NAMES = ["oil_leak", "corrosion"]
MINIMUM_SOURCE_IMAGES = {"oil_leak": 3400, "corrosion": 1200}


def read_names(dataset: Path) -> list[str]:
    config = yaml.safe_load((dataset / "data.yaml").read_text())
    names = config.get("names")
    if isinstance(names, dict):
        return [str(names[index] if index in names else names[str(index)]) for index in range(len(names))]
    if isinstance(names, list):
        return [str(name) for name in names]
    raise ValueError(f"No class names in {dataset / 'data.yaml'}")


def source_class_ids(names: list[str], target: str) -> set[int]:
    matches: set[int] = set()
    for class_id, name in enumerate(names):
        normalized = name.lower().replace("-", "_").replace(" ", "_")
        if target == "oil_leak" and "leak" in normalized:
            matches.add(class_id)
        if target == "corrosion" and ("corrosion" in normalized or "rust" in normalized):
            matches.add(class_id)
    if not matches:
        raise ValueError(f"Could not find {target!r} in source classes {names}")
    return matches


def require_complete_source(source: Path, target: str) -> None:
    count = sum(path.suffix.lower() in IMAGE_SUFFIXES for path in source.rglob("*"))
    minimum = MINIMUM_SOURCE_IMAGES[target]
    if count < minimum:
        raise ValueError(
            f"Incomplete {target} source at {source}: found {count} images, "
            f"expected at least {minimum}. Re-run training/download_datasets.py."
        )


def prepare_output(output: Path, overwrite: bool) -> None:
    if output.exists():
        output_has_contents = any(output.iterdir())
        if output_has_contents and not overwrite:
            raise FileExistsError(f"{output} exists; pass --overwrite to rebuild it")
        if output_has_contents:
            shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)


def valid_label(parts: list[str], path: Path) -> None:
    if len(parts) != 5:
        raise ValueError(f"Invalid YOLO row in {path}: {' '.join(parts)}")
    values = [float(value) for value in parts[1:]]
    if not all(0 <= value <= 1 for value in values) or values[2] <= 0 or values[3] <= 0:
        raise ValueError(f"Invalid normalized box in {path}: {' '.join(parts)}")


def merge_source(
    source: Path,
    output: Path,
    prefix: str,
    target_name: str,
    target_id: int,
) -> dict[str, dict[str, int]]:
    names = read_names(source)
    selected = source_class_ids(names, target_name)
    stats: dict[str, dict[str, int]] = {}
    split_aliases = {"train": "train", "valid": "val", "val": "val", "test": "test"}

    seen_destinations: set[str] = set()
    for source_split, target_split in split_aliases.items():
        images_dir = source / source_split / "images"
        labels_dir = source / source_split / "labels"
        if not images_dir.is_dir() or target_split in seen_destinations:
            continue
        seen_destinations.add(target_split)
        output_images = output / "images" / target_split
        output_labels = output / "labels" / target_split
        output_images.mkdir(parents=True, exist_ok=True)
        output_labels.mkdir(parents=True, exist_ok=True)
        image_count = box_count = negative_count = 0

        for image in sorted(images_dir.iterdir()):
            if image.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            destination_stem = f"{prefix}_{image.stem}"
            label_path = labels_dir / f"{image.stem}.txt"
            rows: list[str] = []
            if label_path.is_file():
                for raw_line in label_path.read_text().splitlines():
                    parts = raw_line.split()
                    if not parts:
                        continue
                    valid_label(parts, label_path)
                    if int(parts[0]) in selected:
                        rows.append(" ".join([str(target_id), *parts[1:]]))
            shutil.copy2(image, output_images / f"{destination_stem}{image.suffix.lower()}")
            (output_labels / f"{destination_stem}.txt").write_text(
                "\n".join(rows) + ("\n" if rows else "")
            )
            image_count += 1
            box_count += len(rows)
            negative_count += not rows

        stats[target_split] = {
            "images": image_count,
            "boxes": box_count,
            "negative_images": negative_count,
        }
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oil", type=Path, default=Path("datasets/sources/oil-leak"))
    parser.add_argument(
        "--corrosion", type=Path, default=Path("datasets/sources/corrosion-bi3q3")
    )
    parser.add_argument("--output", type=Path, default=Path("datasets/hydrovision"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    require_complete_source(args.oil, "oil_leak")
    require_complete_source(args.corrosion, "corrosion")
    try:
        prepare_output(args.output, args.overwrite)
    except FileExistsError as error:
        raise SystemExit(str(error)) from error

    oil_stats = merge_source(args.oil, args.output, "oil", "oil_leak", 0)
    corrosion_stats = merge_source(
        args.corrosion, args.output, "corrosion", "corrosion", 1
    )
    config = {
        "path": str(args.output.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: TARGET_NAMES[0], 1: TARGET_NAMES[1]},
    }
    (args.output / "data.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    manifest = {
        "classes": TARGET_NAMES,
        "sources": {
            "oil_leak": {
                "url": "https://universe.roboflow.com/test-g2mia/leak-kahkr-bqefk/dataset/1",
                "license": "CC BY 4.0",
                "stats": oil_stats,
            },
            "corrosion": {
                "url": "https://universe.roboflow.com/roboflow-100/corrosion-bi3q3/dataset/1",
                "license": "CC BY 4.0",
                "stats": corrosion_stats,
            },
        },
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    print(f"Training config: {args.output / 'data.yaml'}")


if __name__ == "__main__":
    main()
