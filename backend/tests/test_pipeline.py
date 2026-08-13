from __future__ import annotations

import io

import cv2
import numpy as np
from PIL import Image, ImageDraw

from backend.detector import LocalDetector
from backend.main import MAX_LONG_EDGE, VIDEO_SAMPLE_SECONDS, normalize_image


def test_resize_is_enforced() -> None:
    image = Image.new("RGB", (2400, 1200), "#d7d9d4")
    source = io.BytesIO()
    image.save(source, "PNG")
    _, _, width, height = normalize_image(source.getvalue())
    assert max(width, height) == MAX_LONG_EDGE


def test_video_sampling_interval_is_cost_controlled() -> None:
    assert 2 <= VIDEO_SAMPLE_SECONDS <= 3
    assert len(np.arange(0, 30, VIDEO_SAMPLE_SECONDS)) == 12


def test_known_corrosion_patch_returns_bbox_and_confidence() -> None:
    image = Image.new("RGB", (640, 420), "#aeb4b1")
    draw = ImageDraw.Draw(image)
    draw.rectangle((180, 105, 390, 285), fill="#9d3d12")
    frame = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    detections = LocalDetector().predict_batch([frame])[0]
    corrosion = next(item for item in detections if item.defect_type == "corrosion")
    x1, y1, x2, y2 = corrosion.bbox
    assert x1 <= 185 and y1 <= 110 and x2 >= 385 and y2 >= 280
    assert corrosion.confidence >= 0.5
