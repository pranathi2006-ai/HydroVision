#!/usr/bin/env python3
"""Merge source YOLO datasets into HydroVision's two-class schema."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import yaml


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
TARGET_NAMES = ["oil_leak", "corrosion"]
MINIMUM_SOURCE_IMAGES = {
    "oil_leak": 3400,
    "corrosion": 1200,
    "corrosion_rgb_ship": 268,
    "corrosion_rgb_large": 8300,
}


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
            for attempt in range(3):
                try:
                    shutil.rmtree(output)
                    break
                except OSError:
                    leftovers = list(output.iterdir()) if output.exists() else []
                    if leftovers and all(path.name == ".DS_Store" for path in leftovers):
                        for path in leftovers:
                            path.unlink(missing_ok=True)
                        try:
                            output.rmdir()
                            break
                        except OSError:
                            if attempt == 2:
                                raise
                            continue
                    raise
    output.mkdir(parents=True, exist_ok=True)


def replace_output(staging: Path, output: Path) -> None:
    """Replace a completed dataset while retaining rollback on swap failure."""
    backup: Path | None = None
    if output.exists():
        backup = Path(tempfile.mkdtemp(prefix=f".{output.name}-previous-", dir=output.parent))
        backup.rmdir()
        output.rename(backup)
    try:
        staging.rename(output)
    except Exception:
        if backup is not None and backup.exists() and not output.exists():
            backup.rename(output)
        raise
    if backup is not None and backup.exists():
        shutil.rmtree(backup)


def normalized_bbox(parts: list[str], path: Path) -> tuple[int, list[str]]:
    """Return a class id and YOLO bbox from either a box or polygon row."""
    try:
        class_id = int(parts[0])
        values = [float(value) for value in parts[1:]]
    except (IndexError, ValueError) as error:
        raise ValueError(f"Invalid YOLO row in {path}: {' '.join(parts)}") from error

    if len(parts) == 5:
        if not all(0 <= value <= 1 for value in values) or values[2] <= 0 or values[3] <= 0:
            raise ValueError(f"Invalid normalized box in {path}: {' '.join(parts)}")
        return class_id, parts[1:]

    # Roboflow may export instance-segmentation points even when the target
    # HydroVision task is detection. A polygon row is class_id followed by at
    # least three normalized x/y pairs. Convert it to its tight bounding box.
    if len(parts) < 7 or len(values) % 2:
        raise ValueError(f"Invalid YOLO polygon in {path}: {' '.join(parts)}")
    if not all(0 <= value <= 1 for value in values):
        raise ValueError(f"Invalid normalized polygon in {path}: {' '.join(parts)}")
    xs, ys = values[0::2], values[1::2]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    width, height = x_max - x_min, y_max - y_min
    if width <= 0 or height <= 0:
        raise ValueError(f"Degenerate YOLO polygon in {path}: {' '.join(parts)}")
    bbox = (
        (x_min + x_max) / 2,
        (y_min + y_max) / 2,
        width,
        height,
    )
    return class_id, [f"{value:.10g}" for value in bbox]


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
                    class_id, bbox = normalized_bbox(parts, label_path)
                    if class_id in selected:
                        rows.append(" ".join([str(target_id), *bbox]))
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
    parser.add_argument(
        "--corrosion-rgb-ship", type=Path,
        default=Path("datasets/sources/corrosion-rgb-ship"),
    )
    parser.add_argument(
        "--corrosion-rgb-large", type=Path,
        default=Path("datasets/sources/corrosion-rgb-large"),
    )
    parser.add_argument("--skip-corrosion-rgb-large", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("datasets/hydrovision"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    require_complete_source(args.oil, "oil_leak")
    require_complete_source(args.corrosion, "corrosion")
    require_complete_source(args.corrosion_rgb_ship, "corrosion_rgb_ship")
    if not args.skip_corrosion_rgb_large:
        require_complete_source(args.corrosion_rgb_large, "corrosion_rgb_large")
    if args.output.exists() and any(args.output.iterdir()) and not args.overwrite:
        raise SystemExit(f"{args.output} exists; pass --overwrite to rebuild it")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        prefix=f".{args.output.name}-building-", dir=args.output.parent
    ))

    try:
        oil_stats = merge_source(args.oil, staging, "oil", "oil_leak", 0)
        corrosion_stats = merge_source(
            args.corrosion, staging, "corrosion", "corrosion", 1
        )
        corrosion_rgb_ship_stats = merge_source(
            args.corrosion_rgb_ship, staging, "corrosion_rgb_ship", "corrosion", 1
        )
        corrosion_rgb_large_stats = None
        if not args.skip_corrosion_rgb_large:
            corrosion_rgb_large_stats = merge_source(
                args.corrosion_rgb_large, staging, "corrosion_rgb_large", "corrosion", 1
            )
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    config = {
        "path": str(args.output.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: TARGET_NAMES[0], 1: TARGET_NAMES[1]},
    }
    (staging / "data.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    manifest = {
        "classes": TARGET_NAMES,
        "sources": {
            "oil_leak": {
                "url": "https://universe.roboflow.com/test-g2mia/leak-kahkr-bqefk/dataset/1",
                "license": "CC BY 4.0",
                "stats": oil_stats,
            },
            "corrosion_rf100": {
                "url": "https://universe.roboflow.com/roboflow-100/corrosion-bi3q3/dataset/1",
                "license": "CC BY 4.0",
                "stats": corrosion_stats,
            },
            "corrosion_rgb_ship": {
                "url": "https://www.kaggle.com/datasets/wednesday233/corrosion-detect-dataset",
                "license": "MIT",
                "modality": "RGB",
                "stats": corrosion_rgb_ship_stats,
            },
        },
    }
    if corrosion_rgb_large_stats is not None:
        manifest["sources"]["corrosion_rgb_large"] = {
            "url": "https://universe.roboflow.com/averkios/rust-corrosion-detection/dataset/13",
            "license": "CC BY 4.0",
            "modality": "RGB",
            "stats": corrosion_rgb_large_stats,
        }
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    replace_output(staging, args.output)
    print(json.dumps(manifest, indent=2))
    print(f"Training config: {args.output / 'data.yaml'}")


if __name__ == "__main__":
    main()
