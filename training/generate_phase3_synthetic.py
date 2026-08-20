#!/usr/bin/env python3
"""Create deterministic debris or pitting composites from locally licensed images."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def debris_overlay(image: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    mask = np.zeros((height, width), np.uint8)
    center = (int(rng.uniform(.2, .8) * width), int(rng.uniform(.25, .85) * height))
    axes = (int(rng.uniform(.08, .22) * width), int(rng.uniform(.06, .18) * height))
    points = cv2.ellipse2Poly(center, axes, int(rng.integers(0, 180)), 0, 360, 15)
    jitter = rng.integers(-max(2, width // 60), max(3, width // 60), points.shape)
    points = np.clip(points + jitter, (0, 0), (width - 1, height - 1)).astype(np.int32)
    cv2.fillPoly(mask, [points], 255)
    color = np.array([rng.integers(20, 75), rng.integers(45, 105), rng.integers(55, 125)], np.uint8)
    overlay = np.broadcast_to(color, image.shape).copy()
    alpha = cv2.GaussianBlur(mask, (0, 0), 2).astype(np.float32)[:, :, None] / 255 * .82
    result = (image * (1 - alpha) + overlay * alpha).astype(np.uint8)
    return result, mask


def pitting_overlay(image: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    result = image.copy()
    mask = np.zeros(image.shape[:2], np.uint8)
    height, width = mask.shape
    for _ in range(int(rng.integers(25, 80))):
        center = (int(rng.uniform(.1, .9) * width), int(rng.uniform(.1, .9) * height))
        radius = int(rng.integers(max(2, width // 250), max(4, width // 70)))
        cv2.circle(mask, center, radius, 255, -1)
        base = result[center[1], center[0]].astype(np.int16)
        dark = np.clip(base - int(rng.integers(35, 100)), 0, 255).tolist()
        cv2.circle(result, center, radius, dark, -1)
        cv2.circle(result, (center[0] - radius // 3, center[1] - radius // 3), max(1, radius // 4), (190, 190, 190), -1)
    return result, mask


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("debris", "pitting"))
    parser.add_argument("source", type=Path, help="Directory of licensed clear rack or metal backgrounds")
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "masks").mkdir(exist_ok=True)
    rng = np.random.default_rng(args.seed)
    records = []
    transform = debris_overlay if args.mode == "debris" else pitting_overlay
    for source in sorted(path for path in args.source.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES):
        image = cv2.imread(str(source))
        if image is None:
            continue
        result, mask = transform(image, rng)
        digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
        name = f"{args.mode}-{digest}.jpg"
        cv2.imwrite(str(args.output / name), result, [cv2.IMWRITE_JPEG_QUALITY, 92])
        cv2.imwrite(str(args.output / "masks" / f"{Path(name).stem}.png"), mask)
        records.append({
            "image": name,
            "mask": f"masks/{Path(name).stem}.png",
            "source": str(source),
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "generator": "generate_phase3_synthetic.py",
            "seed": args.seed,
        })
    (args.output / "manifest.json").write_text(json.dumps(records, indent=2))
    print(f"Generated {len(records)} {args.mode} composites with masks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
