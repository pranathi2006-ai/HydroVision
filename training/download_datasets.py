#!/usr/bin/env python3
"""Download the two source datasets used for HydroVision training.

Both are public, CC BY 4.0 Roboflow Universe datasets. Roboflow requires a
free private API key to create/download YOLO exports. The key is read from the
environment and never stored.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

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
    print(f"Dataset ready: {destination} ({count} images)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("datasets/sources"))
    parser.add_argument("--skip-oil", action="store_true")
    parser.add_argument("--skip-corrosion", action="store_true")
    args = parser.parse_args()

    if args.skip_oil and args.skip_corrosion:
        print("Nothing to download.")
        return
    api_key = os.getenv("ROBOFLOW_API_KEY", "").strip()
    if not api_key:
        raise SystemExit(
            "ROBOFLOW_API_KEY is required to export the annotated datasets. "
            "Set it in your shell; do not commit it to a file."
        )
    if not args.skip_corrosion:
        download_roboflow_dataset(
            args.output / "corrosion-bi3q3", api_key, "corrosion-bi3q3"
        )
    if not args.skip_oil:
        download_roboflow_dataset(args.output / "oil-leak", api_key, "oil-leak")


if __name__ == "__main__":
    main()
