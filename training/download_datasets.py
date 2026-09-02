#!/usr/bin/env python3
"""Download the source datasets used for HydroVision training.

Roboflow sources require a free private API key to create/download YOLO
exports. The RGB ship-corrosion source is a direct public Kaggle download and
does not require credentials. Keys are read from the environment and never
stored.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import tempfile
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

import yaml
from PIL import Image

DATASETS = {
    "oil-leak": {
        "workspace": "test-g2mia",
        "project": "leak-kahkr-bqefk",
        "version": 1,
        "minimum_images": 3400,
    },
    "corrosion-bi3q3": {
        "workspace": "roboflow-100",
        "project": "corrosion-bi3q3",
        "version": 1,
        "minimum_images": 1200,
    },
    "corrosion-rgb-large": {
        "workspace": "averkios",
        "project": "rust-corrosion-detection",
        "version": 13,
        "minimum_images": 8300,
        "rgb_required": True,
    },
}

SHIP_RGB_DATASET = {
    "url": "https://www.kaggle.com/api/v1/datasets/download/wednesday233/corrosion-detect-dataset",
    "minimum_images": 268,
}


def safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (destination / member.filename).resolve()
            if target != root and root not in target.parents:
                raise RuntimeError(f"Unsafe path in archive: {member.filename}")
        bundle.extractall(destination)


def image_count(destination: Path) -> int:
    suffixes = {".jpg", ".jpeg", ".png", ".webp"}
    return sum(path.suffix.lower() in suffixes for path in destination.rglob("*"))


def verify_rgb_images(destination: Path, maximum_samples: int = 200) -> None:
    images = sorted(
        path for path in destination.rglob("*")
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    if not images:
        raise RuntimeError(f"No images found in {destination}")
    step = max(1, len(images) // maximum_samples)
    sample = images[::step][:maximum_samples]
    non_rgb: list[str] = []
    for path in sample:
        with Image.open(path) as image:
            if not {"R", "G", "B"}.issubset(image.getbands()):
                non_rgb.append(f"{path.name} ({image.mode})")
    if non_rgb:
        raise RuntimeError(
            f"RGB validation failed for {destination}: {', '.join(non_rgb[:5])}"
        )
    print(f"RGB validation passed: {len(sample)} sampled images from {destination}")


def normalize_ship_label(raw_line: str, source: Path) -> tuple[str, bool]:
    parts = raw_line.split()
    if len(parts) != 5 or parts[0] != "1":
        raise RuntimeError(f"Unexpected ship-corrosion label in {source}: {raw_line}")
    x, y, width, height = (float(value) for value in parts[1:])
    repaired = width < 0 or height < 0
    width, height = abs(width), abs(height)
    if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1):
        raise RuntimeError(f"Invalid ship-corrosion box in {source}: {raw_line}")
    return " ".join(("1", f"{x:.8g}", f"{y:.8g}", f"{width:.8g}", f"{height:.8g}")), repaired


def ship_labels_are_valid(destination: Path) -> bool:
    labels = list(destination.rglob("labels/*.txt"))
    if not labels:
        return False
    try:
        for label in labels:
            for line in label.read_text().splitlines():
                if line.strip():
                    normalized, repaired = normalize_ship_label(line, label)
                    if repaired or normalized != line.strip():
                        return False
    except (RuntimeError, ValueError):
        return False
    return True


def download_roboflow_dataset(destination: Path, api_key: str, dataset_name: str) -> None:
    source = DATASETS[dataset_name]
    if (destination / "data.yaml").is_file() and image_count(destination) >= source["minimum_images"]:
        print(f"Dataset already present: {destination}")
        return

    route = (
        f"https://api.roboflow.com/{source['workspace']}/{source['project']}/"
        f"{source['version']}/yolov8"
    )
    request_url = f"{route}?{urllib.parse.urlencode({'api_key': api_key})}"
    try:
        with urllib.request.urlopen(request_url, timeout=60) as response:
            payload = json.load(response)
    except Exception as exc:
        raise RuntimeError(
            f"Roboflow export failed for {dataset_name}. Check ROBOFLOW_API_KEY and "
            "confirm that the Universe dataset is available to your account."
        ) from exc

    export = payload.get("export") or {}
    download_url = export.get("link") if isinstance(export, dict) else None
    if not download_url:
        raise RuntimeError(f"Roboflow did not return an export link: {payload.keys()}")

    archive = destination.parent / f"{dataset_name}-yolov8.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading annotated {dataset_name} data from Roboflow...")
    with urllib.request.urlopen(download_url, timeout=300) as response, archive.open("wb") as output:
        shutil.copyfileobj(response, output)
    safe_extract(archive, destination)
    if not (destination / "data.yaml").is_file():
        candidates = list(destination.rglob("data.yaml"))
        if len(candidates) == 1:
            nested = candidates[0].parent
            for item in nested.iterdir():
                shutil.move(str(item), destination / item.name)
        else:
            raise RuntimeError("Archive did not contain one recognizable YOLO dataset")
    count = image_count(destination)
    if count < source["minimum_images"]:
        raise RuntimeError(
            f"Incomplete {dataset_name} export: found {count} images, expected at least "
            f"{source['minimum_images']}"
        )
    if source.get("rgb_required"):
        verify_rgb_images(destination)
    print(f"Dataset ready: {destination} ({count} images)")


def download_ship_rgb_dataset(destination: Path) -> None:
    """Download and deterministically split the flat Kaggle YOLO collection."""
    if (
        (destination / "data.yaml").is_file()
        and image_count(destination) >= SHIP_RGB_DATASET["minimum_images"]
        and ship_labels_are_valid(destination)
    ):
        verify_rgb_images(destination, maximum_samples=SHIP_RGB_DATASET["minimum_images"])
        print(f"Dataset already present: {destination}")
        return

    archive = destination.parent / "corrosion-rgb-ship.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.is_file() or not zipfile.is_zipfile(archive):
        print("Downloading annotated RGB ship-corrosion data from Kaggle...")
        request = urllib.request.Request(
            SHIP_RGB_DATASET["url"],
            headers={"User-Agent": "HydroVision-dataset-downloader/1.0"},
        )
        with urllib.request.urlopen(request, timeout=300) as response, archive.open("wb") as output:
            shutil.copyfileobj(response, output)
    else:
        print(f"Using existing archive: {archive}")

    with tempfile.TemporaryDirectory(prefix="hydrovision-ship-rgb-", dir=destination.parent) as temporary:
        extracted = Path(temporary) / "extracted"
        safe_extract(archive, extracted)
        image_dirs = [path for path in extracted.rglob("images") if path.is_dir()]
        label_dirs = [path for path in extracted.rglob("labels") if path.is_dir()]
        if len(image_dirs) != 1 or len(label_dirs) != 1:
            raise RuntimeError("Ship-corrosion archive did not contain one images/labels pair")
        images_dir, labels_dir = image_dirs[0], label_dirs[0]
        images = sorted(
            path for path in images_dir.iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        )
        groups: dict[str, list[Path]] = {}
        for image in images:
            group = re.sub(r"_\d+$", "", image.stem)
            groups.setdefault(group, []).append(image)
        group_names = sorted(groups)
        random.Random(42).shuffle(group_names)
        train_end = round(len(group_names) * 0.70)
        val_end = round(len(group_names) * 0.90)
        assignments = {
            group: "train" if index < train_end else "valid" if index < val_end else "test"
            for index, group in enumerate(group_names)
        }

        prepared = Path(temporary) / "prepared"
        repaired_boxes = 0
        for group, grouped_images in groups.items():
            split = assignments[group]
            for image in grouped_images:
                label = labels_dir / f"{image.stem}.txt"
                if not label.is_file():
                    raise RuntimeError(f"Missing YOLO label for {image.name}")
                normalized_rows: list[str] = []
                for line in label.read_text().splitlines():
                    if line.strip():
                        normalized, repaired = normalize_ship_label(line, label)
                        normalized_rows.append(normalized)
                        repaired_boxes += repaired
                target_images = prepared / split / "images"
                target_labels = prepared / split / "labels"
                target_images.mkdir(parents=True, exist_ok=True)
                target_labels.mkdir(parents=True, exist_ok=True)
                shutil.copy2(image, target_images / image.name)
                (target_labels / label.name).write_text(
                    "\n".join(normalized_rows) + ("\n" if normalized_rows else "")
                )
        (prepared / "data.yaml").write_text(yaml.safe_dump({
            "train": "../train/images",
            "val": "../valid/images",
            "test": "../test/images",
            "names": {0: "unused", 1: "corrosion"},
        }, sort_keys=False))
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(prepared, destination)

    if repaired_boxes:
        print(f"Repaired {repaired_boxes} negative-width/height YOLO boxes from the source archive")

    count = image_count(destination)
    if count < SHIP_RGB_DATASET["minimum_images"]:
        raise RuntimeError(f"Incomplete RGB ship-corrosion dataset: found {count} images")
    verify_rgb_images(destination, maximum_samples=SHIP_RGB_DATASET["minimum_images"])
    print(f"Dataset ready: {destination} ({count} images)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("datasets/sources"))
    parser.add_argument("--skip-oil", action="store_true")
    parser.add_argument("--skip-corrosion", action="store_true")
    parser.add_argument("--skip-corrosion-rgb-ship", action="store_true")
    parser.add_argument("--skip-corrosion-rgb-large", action="store_true")
    args = parser.parse_args()

    if all((args.skip_oil, args.skip_corrosion, args.skip_corrosion_rgb_ship, args.skip_corrosion_rgb_large)):
        print("Nothing to download.")
        return
    if not args.skip_corrosion_rgb_ship:
        download_ship_rgb_dataset(args.output / "corrosion-rgb-ship")

    api_key = os.getenv("ROBOFLOW_API_KEY", "").strip()
    needs_roboflow = not args.skip_oil or not args.skip_corrosion or not args.skip_corrosion_rgb_large
    if needs_roboflow and not api_key:
        raise SystemExit(
            "ROBOFLOW_API_KEY is required for the selected Roboflow exports. "
            "The credential-free RGB ship dataset was prepared first. Set the key in "
            "your shell or skip the Roboflow sources; do not commit the key to a file."
        )
    if not args.skip_corrosion:
        download_roboflow_dataset(
            args.output / "corrosion-bi3q3", api_key, "corrosion-bi3q3"
        )
    if not args.skip_oil:
        download_roboflow_dataset(args.output / "oil-leak", api_key, "oil-leak")
    if not args.skip_corrosion_rgb_large:
        download_roboflow_dataset(
            args.output / "corrosion-rgb-large", api_key, "corrosion-rgb-large"
        )


if __name__ == "__main__":
    main()
