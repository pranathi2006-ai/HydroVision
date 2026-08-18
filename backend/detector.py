from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


@dataclass(frozen=True)
class Detection:
    defect_type: str
    confidence: float
    bbox: tuple[int, int, int, int]
    affected_area_pct: float


class LocalDetector:
    """Runs a local YOLO model when weights exist, with an offline CV fallback.

    The fallback keeps the MVP usable before fine-tuned weights are supplied and
    is intentionally conservative. It detects contiguous rust-brown regions and
    dark, glossy leak-shaped regions; it never sends pixels off the machine.
    """

    def __init__(self, model_path: str | None = None) -> None:
        self.model = None
        repository_root = Path(__file__).resolve().parents[1]
        default_model = repository_root / "models" / "hydrovision-yolov8n.pt"
        configured = model_path or os.getenv("HYDROVISION_MODEL_PATH")
        if not configured and default_model.is_file():
            configured = str(default_model)
        if configured and Path(configured).is_file():
            from ultralytics import YOLO

            self.model = YOLO(configured)

    @property
    def engine_name(self) -> str:
        return "local-yolo" if self.model else "local-cv-baseline"

    def predict_batch(self, images: Iterable[np.ndarray]) -> list[list[Detection]]:
        batch = list(images)
        if not batch:
            return []
        if self.model:
            return self._predict_yolo(batch)
        return [self._predict_baseline(image) for image in batch]

    def _predict_yolo(self, images: list[np.ndarray]) -> list[list[Detection]]:
        results = self.model.predict(
            source=images,
            imgsz=1024,
            conf=0.25,
            device="cpu",
            verbose=False,
            stream=False,
        )
        output: list[list[Detection]] = []
        for image, result in zip(images, results):
            height, width = image.shape[:2]
            image_area = max(1, width * height)
            detections: list[Detection] = []
            for box in result.boxes:
                x1, y1, x2, y2 = (int(value) for value in box.xyxy[0].tolist())
                raw_name = str(result.names[int(box.cls[0])]).lower()
                defect_type = self._canonical_defect_type(raw_name)
                if defect_type is None:
                    continue
                area = max(0, x2 - x1) * max(0, y2 - y1)
                detections.append(
                    Detection(
                        defect_type=defect_type,
                        confidence=round(float(box.conf[0]), 4),
                        bbox=(x1, y1, x2, y2),
                        affected_area_pct=round(area / image_area * 100, 2),
                    )
                )
            output.append(detections)
        return output

    @staticmethod
    def _canonical_defect_type(raw_name: str) -> str | None:
        normalized = raw_name.lower().replace("-", "_").replace(" ", "_")
        if "corrosion" in normalized or "rust" in normalized:
            return "corrosion"
        if "leak" in normalized and ("oil" in normalized or normalized == "leak"):
            return "leak"
        return None

    def _predict_baseline(self, image: np.ndarray) -> list[Detection]:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        rust_mask = cv2.inRange(hsv, np.array([3, 70, 35]), np.array([28, 255, 230]))

        # Oil/water leaks are approximated as dark, saturated pools with bright
        # specular pixels nearby. This is a deterministic baseline, not a claim
        # of production-grade accuracy.
        dark = cv2.inRange(hsv, np.array([0, 30, 0]), np.array([179, 255, 90]))
        value = hsv[:, :, 2]
        highlights = cv2.inRange(value, 205, 255)
        highlights = cv2.dilate(highlights, np.ones((17, 17), np.uint8))
        leak_mask = cv2.bitwise_and(dark, highlights)

        kernel = np.ones((7, 7), np.uint8)
        rust_mask = cv2.morphologyEx(rust_mask, cv2.MORPH_CLOSE, kernel)
        leak_mask = cv2.morphologyEx(leak_mask, cv2.MORPH_CLOSE, kernel)

        detections = self._regions(rust_mask, "corrosion", image.shape)
        detections.extend(self._regions(leak_mask, "leak", image.shape))
        return sorted(detections, key=lambda item: item.confidence, reverse=True)[:12]

    @staticmethod
    def _regions(mask: np.ndarray, defect_type: str, shape: tuple[int, ...]) -> list[Detection]:
        height, width = shape[:2]
        image_area = max(1, width * height)
        minimum_area = max(120, int(image_area * 0.0012))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections: list[Detection] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < minimum_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            coverage = area / image_area * 100
            fill_ratio = area / max(1, w * h)
            confidence = min(0.94, 0.42 + min(0.32, coverage / 18) + fill_ratio * 0.22)
            detections.append(
                Detection(
                    defect_type=defect_type,
                    confidence=round(confidence, 4),
                    bbox=(x, y, x + w, y + h),
                    affected_area_pct=round(coverage, 2),
                )
            )
        return detections
