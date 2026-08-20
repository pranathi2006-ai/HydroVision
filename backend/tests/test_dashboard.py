from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from backend.attribution import AttributionService, AttributionSettings
from backend.performance import PerformanceReading
from backend.reference_curves import (
    PerformanceCalculationService,
    PerformanceCurveModel,
    import_reference_curves,
)
from backend.store import Store


UTC = timezone.utc
MOCK_CURVES = Path(__file__).resolve().parents[2] / "reference_curves" / "mock_design"


def services(tmp_path: Path):
    store = Store(tmp_path / "hydrovision.sqlite3")
    import_reference_curves(store, MOCK_CURVES, dataset_name="dashboard curves", is_demo=True)
    calculator = PerformanceCalculationService(
        store, PerformanceCurveModel(store, "turbine_1"), nameplate_capacity_mw=75,
    )
    attribution = AttributionService(
        store, calculator.curves,
        AttributionSettings(threshold_pct=5, evidence_window_seconds=21600),
    )
    return store, calculator, attribution


def test_current_dashboard_is_one_six_site_read_model_with_linked_evidence(tmp_path: Path) -> None:
    store, calculator, attribution = services(tmp_path)
    reading_ts = datetime(2026, 8, 20, 10, 0, tzinfo=UTC)
    media_id = store.insert_media(
        content_hash="dashboard-rack-evidence",
        original_name="rack.jpg",
        stored_name="rack.jpg",
        media_type="image",
        location_id="intake",
        width=640,
        height=480,
        inference_engine="test",
    )
    event = store.insert_detection_events(
        asset_id="intake_gate",
        sensor_id="intake_gate_camera",
        media_id=media_id,
        cache_key="dashboard-rack-event",
        events=[{
            "ts": (reading_ts - timedelta(minutes=5)).isoformat(),
            "detection_type": "trash_rack_blockage",
            "defect_present": True,
            "severity": "critical",
            "confidence": 0.96,
            "bbox": [20, 30, 200, 220],
            "measurement": {"blockage_pct": 35.0},
            "engine": "test",
        }],
    )[0]
    reading = PerformanceReading(reading_ts, 118.0, 82.0, 60.0, 40.0)
    reading_id = store.insert_performance_reading(reading, calculator.calculate(reading))
    attribution.attribute_reading(reading_id)

    dashboard = store.current_dashboard()

    assert dashboard["reading"]["reading_id"] == reading_id
    assert dashboard["attribution_status"] == "attributed"
    assert len(dashboard["sites"]) == 6
    assert dashboard["sites"][0]["asset_id"] == "intake_gate"
    intake = dashboard["sites"][0]
    assert intake["latest_event"]["event_id"] == event["event_id"]
    assert intake["latest_event"]["measurement"]["blockage_pct"] == 35.0
    assert intake["latest_event"]["thumbnail_url"] == "/api/media/rack.jpg"
    assert intake["attribution"]["event_id"] == event["event_id"]
    assert intake["attribution"]["estimated_loss_mw"] > 0
    assert "Schedule rack cleaning" in intake["recommended_action"]
    assert all(site["asset_id"] != "main_transformer" or site["attribution"] is None for site in dashboard["sites"])


def test_dashboard_exposes_unexplained_current_gap_without_fake_site_rows(tmp_path: Path) -> None:
    store, calculator, attribution = services(tmp_path)
    reading_ts = datetime(2026, 8, 20, 10, 0, tzinfo=UTC)
    reading = PerformanceReading(reading_ts, 118.0, 82.0, 60.0, 40.0)
    reading_id = store.insert_performance_reading(reading, calculator.calculate(reading))
    attribution.attribute_reading(reading_id)

    dashboard = store.current_dashboard()

    assert dashboard["attribution_status"] == "unexplained"
    assert all(site["attribution"] is None for site in dashboard["sites"])


def test_dashboard_without_readings_still_returns_all_sites_and_actions(tmp_path: Path) -> None:
    store = Store(tmp_path / "hydrovision.sqlite3")

    dashboard = store.current_dashboard()

    assert dashboard["reading"] is None
    assert dashboard["attribution_status"] == "not_triggered"
    assert {site["asset_id"] for site in dashboard["sites"]} == {
        "turbine_1", "turbine_2", "penstock_valve", "main_transformer",
        "intake_gate", "draft_tube",
    }
    with store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM recommended_action").fetchone()[0] == 7
