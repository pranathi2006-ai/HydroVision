#!/usr/bin/env python3
"""Train, evaluate, and export the HydroVision two-class detector."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO


def automatic_device() -> str:
    if torch.cuda.is_available():
        return "0"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("datasets/hydrovision/data.yaml"))
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=automatic_device())
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--name", default="oil-leak-corrosion-yolov8n")
    parser.add_argument(
        "--output", type=Path, default=Path("models/hydrovision-yolov8n.pt")
    )
    args = parser.parse_args()

    if not args.data.is_file():
        raise SystemExit(f"Dataset config not found: {args.data}")
    model = YOLO(args.model)
    result = model.train(
        data=str(args.data.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(Path("runs/hydrovision").resolve()),
        name=args.name,
        patience=20,
        seed=42,
        deterministic=True,
        cache="disk",
        plots=True,
    )
    best = Path(result.save_dir) / "weights" / "best.pt"
    if not best.is_file():
        raise RuntimeError(f"Training completed without {best}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, args.output)
    deployed = YOLO(str(args.output))
    metrics = deployed.val(data=str(args.data.resolve()), split="test", device=args.device)
    summary = {
        "weights": str(args.output.resolve()),
        "classes": deployed.names,
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
    }
    metrics_path = args.output.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
