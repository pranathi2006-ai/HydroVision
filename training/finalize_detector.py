#!/usr/bin/env python3
"""Validate a trained detector on CPU, then promote it for backend inference."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

# Keep Matplotlib's generated cache inside the project rather than attempting
# to write to a restricted user-home directory during CPU validation.
matplotlib_cache = Path("runs/.matplotlib").resolve()
matplotlib_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("runs/hydrovision/hydrovision-rgb-corrosion/weights/best.pt"),
    )
    parser.add_argument(
        "--data", type=Path, default=Path("datasets/hydrovision/data.yaml")
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument(
        "--output", type=Path, default=Path("models/hydrovision-yolov8n.pt")
    )
    parser.add_argument("--name", default="rgb-corrosion-cpu-test")
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")
    if not args.data.is_file():
        raise SystemExit(f"Dataset config not found: {args.data}")

    model = YOLO(str(args.checkpoint))
    metrics = model.val(
        data=str(args.data.resolve()),
        split=args.split,
        device=args.device,
        project=str(Path("runs/hydrovision").resolve()),
        name=args.name,
        exist_ok=True,
        plots=True,
    )

    per_class = {}
    for class_id, class_name in model.names.items():
        per_class[class_name] = {
            "precision": float(metrics.box.p[class_id]),
            "recall": float(metrics.box.r[class_id]),
            "map50": float(metrics.box.ap50[class_id]),
            "map50_95": float(metrics.box.maps[class_id]),
        }
    summary = {
        "validated_checkpoint": str(args.checkpoint.resolve()),
        "validation_device": args.device,
        "validation_split": args.split,
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "per_class": per_class,
    }

    # Promote only after a complete validation run, so a failed validation
    # cannot replace the last known working application model.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.checkpoint, args.output)
    metrics_path = args.output.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Promoted validated model: {args.output}")


if __name__ == "__main__":
    main()
