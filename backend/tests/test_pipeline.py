from __future__ import annotations

import io
import json
import re
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi import HTTPException, Response
from PIL import Image, ImageDraw

import backend.main as main_api
from backend.detector import LocalDetector
from backend.main import (
    LAN_ORIGIN_REGEX,
    MAX_LONG_EDGE,
    VIDEO_SAMPLE_SECONDS,
    current_dashboard,
    normalize_image,
    performance_settings,
)
from backend.store import Store


def test_resize_is_enforced() -> None:
    image = Image.new("RGB", (2400, 1200), "#d7d9d4")
    source = io.BytesIO()
    image.save(source, "PNG")
    _, _, width, height = normalize_image(source.getvalue())
    assert max(width, height) == MAX_LONG_EDGE


def test_video_sampling_interval_is_cost_controlled() -> None:
    assert 2 <= VIDEO_SAMPLE_SECONDS <= 3
    assert len(np.arange(0, 30, VIDEO_SAMPLE_SECONDS)) == 12


def test_private_network_origins_are_allowed_without_open_public_cors() -> None:
    assert re.fullmatch(LAN_ORIGIN_REGEX, "http://192.168.1.42:3000")
    assert re.fullmatch(LAN_ORIGIN_REGEX, "http://10.0.0.8:3000")
    assert not re.fullmatch(LAN_ORIGIN_REGEX, "https://example.com")


def test_current_dashboard_endpoint_uses_ingestion_cadence_and_six_sites() -> None:
    response = Response()

    payload = current_dashboard(response)

    assert response.headers["X-Poll-Interval-Seconds"] == str(
        performance_settings.effective_interval_seconds
    )
    assert payload["poll_interval_seconds"] == performance_settings.effective_interval_seconds
    assert len(payload["sites"]) == 6


def test_engineer_feedback_endpoint_captures_one_confirmed_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    local_store = Store(tmp_path / "hydrovision.sqlite3")
    with local_store.connect() as db:
        reading_id = db.execute(
            """
            INSERT INTO performance_reading (
                ts, headwater_level, tailwater_level, gate_position,
                theoretical_mw, actual_mw, gap_pct
            ) VALUES ('2026-08-20T10:00:00+00:00', 118, 82, 60, 50, 42, 16)
            """
        ).lastrowid
        event_id = db.execute(
            """
            INSERT INTO detection_event (
                ts, asset_id, sensor_id, detection_type, defect_present,
                severity, confidence, measurement, engine, cache_key, created_at
            ) VALUES ('2026-08-20T09:55:00+00:00', 'penstock_valve',
                      'penstock_valve_camera', 'oil_leak', 1, 'critical', 0.9,
                      ?, 'test', 'feedback-endpoint-event', '2026-08-20T09:55:00+00:00')
            """,
            (json.dumps({"affected_area_pct": 12.0}),),
        ).lastrowid
        attribution_id = db.execute(
            """
            INSERT INTO loss_attribution (
                reading_id, asset_id, event_id, estimated_loss_mw,
                confidence, method, rule_estimate_mw, rule_confidence
            ) VALUES (?, 'penstock_valve', ?, 1.5, 0.8, 'rule_based', 1.5, 0.8)
            """,
            (reading_id, event_id),
        ).lastrowid
    monkeypatch.setattr(main_api, "store", local_store)
    payload = main_api.AttributionFeedbackRequest(
        confirmed=True,
        notes="Leak repair recovered output.",
        confirmed_by="engineer.name",
    )

    feedback = main_api.attribution_feedback(int(attribution_id), payload)

    assert feedback["confirmed"] is True
    assert feedback["confirmed_by"] == "engineer.name"
    with pytest.raises(HTTPException) as duplicate:
        main_api.attribution_feedback(int(attribution_id), payload)
    assert duplicate.value.status_code == 409


def test_known_corrosion_patch_returns_bbox_and_confidence() -> None:
    image = Image.new("RGB", (640, 420), "#aeb4b1")
    draw = ImageDraw.Draw(image)
    draw.rectangle((180, 105, 390, 285), fill="#9d3d12")
    frame = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    # This fixture validates the deterministic CV baseline, not trained weights.
    detections = LocalDetector(model_path="/definitely/missing/hydrovision-test-model.pt").predict_batch([frame])[0]
    corrosion = next(item for item in detections if item.defect_type == "corrosion")
    x1, y1, x2, y2 = corrosion.bbox
    assert x1 <= 185 and y1 <= 110 and x2 >= 385 and y2 >= 280
    assert corrosion.confidence >= 0.5


def test_model_class_names_are_mapped_conservatively() -> None:
    assert LocalDetector._canonical_defect_type("oil_leak") == "leak"
    assert LocalDetector._canonical_defect_type("OIL LEAKAGE") == "leak"
    assert LocalDetector._canonical_defect_type("rust") == "corrosion"
    assert LocalDetector._canonical_defect_type("corrosion") == "corrosion"
    assert LocalDetector._canonical_defect_type("crack") is None
