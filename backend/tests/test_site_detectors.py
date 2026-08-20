from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np

from backend.detector import LocalDetector
from backend.site_detectors import (
    AreaBlockageDetector,
    GatePositionVerifier,
    ThermalHotspotDetector,
    SiteDetectionService,
)
from backend.store import Store


def test_exact_six_assets_and_required_sensors_are_seeded(tmp_path: Path) -> None:
    store = Store(tmp_path / "hydrovision.sqlite3")
    rows = store.site_assets()
    assert {row["asset_id"] for row in rows} == {
        "turbine_1", "turbine_2", "penstock_valve", "main_transformer",
        "intake_gate", "draft_tube",
    }
    sensors = {(row["asset_id"], row["sensor_type"]) for row in rows}
    assert all((asset_id, "camera") in sensors for asset_id in {
        "turbine_1", "turbine_2", "penstock_valve", "main_transformer",
        "intake_gate", "draft_tube",
    })
    assert ("main_transformer", "thermal_camera") in sensors


def test_blockage_detector_measures_synthetic_debris_area() -> None:
    image = np.full((300, 500, 3), 180, np.uint8)
    cv2.rectangle(image, (120, 80), (360, 240), (35, 75, 95), -1)
    event = AreaBlockageDetector("trash_rack_blockage", 8, "MISSING_MODEL_ENV").predict(image)
    assert event.defect_present
    assert 20 <= event.measurement["blockage_pct"] <= 35
    assert event.bbox is not None


def test_gate_geometry_flags_position_mismatch(monkeypatch) -> None:
    monkeypatch.setenv("HYDROVISION_GATE_MISMATCH_THRESHOLD_PCT", "8")
    image = np.full((400, 600, 3), 170, np.uint8)
    cv2.line(image, (50, 80), (550, 80), (15, 15, 15), 8)
    event = GatePositionVerifier().predict(image, commanded_pct=20)
    assert event.measurement["geometry_valid"]
    assert event.measurement["visual_gate_position_pct"] >= 95
    assert event.defect_present


def test_thermal_rule_requires_three_consecutive_frames(monkeypatch) -> None:
    monkeypatch.setenv("HYDROVISION_THERMAL_PERSISTENCE_FRAMES", "3")
    detector = ThermalHotspotDetector()
    image = np.full((240, 320, 3), 85, np.uint8)
    cv2.rectangle(image, (120, 80), (200, 160), (250, 250, 250), -1)
    first = detector.predict(image, "main_transformer")
    second = detector.predict(image, "main_transformer")
    third = detector.predict(image, "main_transformer")
    assert not first.defect_present and not second.defect_present
    assert third.defect_present
    assert third.measurement["delta_t_c"] > 12


def _insert_media(store: Store, content_hash: str, location_id: str) -> int:
    return store.insert_media(
        content_hash=content_hash,
        original_name=f"{content_hash}.jpg",
        stored_name=f"{content_hash}.jpg",
        media_type="image",
        location_id=location_id,
        width=320,
        height=240,
        inference_engine="test",
    )


def test_every_site_yields_explicit_event_and_hash_cache_is_reused(tmp_path: Path) -> None:
    store = Store(tmp_path / "hydrovision.sqlite3")
    service = SiteDetectionService(
        store,
        LocalDetector(model_path="/definitely/missing/phase3-test-model.pt"),
    )
    image = np.full((240, 320, 3), 180, np.uint8)
    routes = [
        ("turbine_1", "turbine_1_camera", "turbine-a"),
        ("turbine_2", "turbine_2_camera", "turbine-b"),
        ("penstock_valve", "penstock_valve_camera", "penstock"),
        ("main_transformer", "main_transformer_thermal", "transformer"),
        ("intake_gate", "intake_gate_camera", "intake"),
        ("draft_tube", "draft_tube_camera", "draft-tube"),
    ]
    first_events = None
    for index, (asset_id, sensor_id, location_id) in enumerate(routes):
        content_hash = hashlib.sha256(f"site-{index}".encode()).hexdigest()
        media_id = _insert_media(store, content_hash, location_id)
        events = service.analyze(
            image=image, media_id=media_id, content_hash=content_hash,
            asset_id=asset_id, sensor_id=sensor_id,
        )
        assert events
        assert all("defect_present" in event and event["measurement"] for event in events)
        if index == 0:
            first_events = events
            repeated = service.analyze(
                image=image, media_id=media_id, content_hash=content_hash,
                asset_id=asset_id, sensor_id=sensor_id,
            )
            assert repeated == first_events

    assert {event["asset_id"] for event in store.detection_events()} == {
        route[0] for route in routes
    }
