from __future__ import annotations

import logging
import json
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


def setup_services(tmp_path: Path, *, meter_location: str = "generator_terminal"):
    store = Store(tmp_path / "hydrovision.sqlite3")
    import_reference_curves(store, MOCK_CURVES, dataset_name="test curves", is_demo=True)
    calculator = PerformanceCalculationService(
        store, PerformanceCurveModel(store, "turbine_1"), nameplate_capacity_mw=75,
    )
    attribution = AttributionService(
        store,
        calculator.curves,
        AttributionSettings(
            threshold_pct=5,
            evidence_window_seconds=6 * 60 * 60,
            meter_location=meter_location,
        ),
    )
    return store, calculator, attribution


def insert_reading(store: Store, calculator: PerformanceCalculationService, ts: datetime, actual_mw: float) -> int:
    reading = PerformanceReading(ts, 118.0, 82.0, 60.0, actual_mw)
    return store.insert_performance_reading(reading, calculator.calculate(reading))


def insert_event(
    store: Store,
    *,
    ts: datetime,
    asset_id: str,
    sensor_id: str,
    detection_type: str,
    severity: str,
    confidence: float,
    measurement: dict,
    cache_key: str,
    defect_present: bool = True,
) -> int:
    rows = store.insert_detection_events(
        asset_id=asset_id,
        sensor_id=sensor_id,
        media_id=None,
        cache_key=cache_key,
        events=[{
            "ts": ts.isoformat(),
            "detection_type": detection_type,
            "defect_present": defect_present,
            "severity": severity,
            "confidence": confidence,
            "bbox": None,
            "measurement": measurement,
            "engine": "test-rule-evidence",
        }],
    )
    return int(rows[0]["event_id"])


def test_known_trash_rack_blockage_is_geometrically_attributed(tmp_path: Path) -> None:
    store, calculator, service = setup_services(tmp_path)
    reading_ts = datetime(2026, 8, 20, 10, 0, tzinfo=UTC)
    event_id = insert_event(
        store,
        ts=reading_ts - timedelta(minutes=10),
        asset_id="intake_gate",
        sensor_id="intake_gate_camera",
        detection_type="trash_rack_blockage",
        severity="critical",
        confidence=0.96,
        measurement={"blockage_pct": 40.0, "status": "blocked"},
        cache_key="trash-known-cleaning",
    )
    reading_id = insert_reading(store, calculator, reading_ts, actual_mw=40)

    result = service.attribute_reading(reading_id)
    ranked = store.loss_attributions(reading_id)

    assert result == {"reading_id": reading_id, "status": "attributed", "created": 1}
    assert len(ranked) == 1
    assert ranked[0]["asset_id"] == "intake_gate"
    assert ranked[0]["event_id"] == event_id
    assert ranked[0]["estimated_loss_mw"] > 1
    assert ranked[0]["confidence"] > 0.8
    assert ranked[0]["method"] == "rule_based"


def test_below_threshold_never_creates_attribution_run(tmp_path: Path) -> None:
    store, calculator, service = setup_services(tmp_path)
    reading_id = insert_reading(
        store, calculator, datetime(2026, 8, 20, 10, 0, tzinfo=UTC), actual_mw=52,
    )

    result = service.attribute_reading(reading_id)

    assert result["status"] == "below_threshold"
    assert store.attribution_run(reading_id) is None
    assert store.loss_attributions(reading_id) == []


def test_processed_reading_is_deduplicated(tmp_path: Path) -> None:
    store, calculator, service = setup_services(tmp_path)
    reading_ts = datetime(2026, 8, 20, 10, 0, tzinfo=UTC)
    insert_event(
        store, ts=reading_ts - timedelta(minutes=5), asset_id="penstock_valve",
        sensor_id="penstock_valve_camera", detection_type="oil_leak",
        severity="critical", confidence=0.9,
        measurement={"affected_area_pct": 15}, cache_key="one-oil-event",
    )
    reading_id = insert_reading(store, calculator, reading_ts, actual_mw=40)

    first = service.attribute_reading(reading_id)
    second = service.attribute_reading(reading_id)

    assert first["created"] == 1
    assert second["created"] == 0 and second["deduplicated"] is True
    assert len(store.loss_attributions(reading_id)) == 1


def test_gap_without_recent_active_evidence_is_logged_unexplained(
    tmp_path: Path, caplog,
) -> None:
    store, calculator, service = setup_services(tmp_path)
    reading_id = insert_reading(
        store, calculator, datetime(2026, 8, 20, 10, 0, tzinfo=UTC), actual_mw=40,
    )

    with caplog.at_level(logging.WARNING, logger="hydrovision.performance.attribution"):
        result = service.attribute_reading(reading_id)

    assert result["status"] == "unexplained"
    assert store.attribution_run(reading_id)["status"] == "unexplained"
    assert store.loss_attributions(reading_id) == []
    assert "gap attribution unexplained" in caplog.text


def test_transformer_event_is_never_used_for_generation_gap(tmp_path: Path) -> None:
    store, calculator, service = setup_services(tmp_path)
    reading_ts = datetime(2026, 8, 20, 10, 0, tzinfo=UTC)
    insert_event(
        store, ts=reading_ts - timedelta(minutes=5), asset_id="main_transformer",
        sensor_id="main_transformer_thermal", detection_type="thermal_hotspot",
        severity="critical", confidence=0.99,
        measurement={"delta_t_c": 40}, cache_key="transformer-hot",
    )
    reading_id = insert_reading(store, calculator, reading_ts, actual_mw=40)

    assert service.attribute_reading(reading_id)["status"] == "unexplained"
    assert store.loss_attributions(reading_id) == []


def test_grid_connection_or_unconfirmed_meter_disables_attribution(tmp_path: Path) -> None:
    store, calculator, service = setup_services(tmp_path, meter_location="grid_connection")
    reading_id = insert_reading(
        store, calculator, datetime(2026, 8, 20, 10, 0, tzinfo=UTC), actual_mw=40,
    )

    assert service.attribute_reading(reading_id)["status"] == "meter_location_unconfirmed"
    assert store.attribution_run(reading_id) is None


def test_real_source_requires_explicit_meter_confirmation(monkeypatch) -> None:
    monkeypatch.delenv("HYDROVISION_ACTUAL_MW_METER_LOCATION", raising=False)
    assert AttributionSettings.from_env("mock").meter_location == "generator_terminal"
    real = AttributionSettings.from_env("real")
    assert real.meter_location == "unconfirmed"
    assert not real.enabled


def test_gate_rule_uses_visual_position_and_keeps_event_link(tmp_path: Path) -> None:
    store, calculator, service = setup_services(tmp_path)
    reading_ts = datetime(2026, 8, 20, 10, 0, tzinfo=UTC)
    event_id = insert_event(
        store, ts=reading_ts - timedelta(minutes=5), asset_id="intake_gate",
        sensor_id="intake_gate_camera", detection_type="gate_position_mismatch",
        severity="warning", confidence=0.94,
        measurement={
            "visual_gate_position_pct": 48,
            "commanded_gate_position_pct": 60,
            "mismatch_pct_points": 12,
        },
        cache_key="gate-under-open",
    )
    reading_id = insert_reading(store, calculator, reading_ts, actual_mw=40)

    service.attribute_reading(reading_id)
    ranked = store.loss_attributions(reading_id)

    assert ranked[0]["event_id"] == event_id
    assert ranked[0]["estimated_loss_mw"] > 0
    assert ranked[0]["confidence"] > 0.8


def test_only_closest_prior_event_per_asset_and_type_is_used(tmp_path: Path) -> None:
    store, calculator, service = setup_services(tmp_path)
    reading_ts = datetime(2026, 8, 20, 10, 0, tzinfo=UTC)
    insert_event(
        store, ts=reading_ts - timedelta(hours=2), asset_id="penstock_valve",
        sensor_id="penstock_valve_camera", detection_type="oil_leak",
        severity="warning", confidence=0.8, measurement={}, cache_key="older-leak",
    )
    latest_event = insert_event(
        store, ts=reading_ts - timedelta(minutes=5), asset_id="penstock_valve",
        sensor_id="penstock_valve_camera", detection_type="oil_leak",
        severity="critical", confidence=0.9, measurement={}, cache_key="newer-leak",
    )
    reading_id = insert_reading(store, calculator, reading_ts, actual_mw=40)

    service.attribute_reading(reading_id)

    ranked = store.loss_attributions(reading_id)
    assert len(ranked) == 1
    assert ranked[0]["event_id"] == latest_event
    assert all(row["event_id"] is not None for row in ranked)


def test_newer_healthy_event_resolves_older_defect(tmp_path: Path) -> None:
    store, calculator, service = setup_services(tmp_path)
    reading_ts = datetime(2026, 8, 20, 10, 0, tzinfo=UTC)
    insert_event(
        store, ts=reading_ts - timedelta(hours=2), asset_id="penstock_valve",
        sensor_id="penstock_valve_camera", detection_type="oil_leak",
        severity="critical", confidence=0.9, measurement={}, cache_key="resolved-old-leak",
    )
    insert_event(
        store, ts=reading_ts - timedelta(minutes=5), asset_id="penstock_valve",
        sensor_id="penstock_valve_camera", detection_type="oil_leak",
        severity="normal", confidence=0.9, measurement={"status": "healthy"},
        cache_key="resolved-healthy", defect_present=False,
    )
    reading_id = insert_reading(store, calculator, reading_ts, actual_mw=40)

    assert service.attribute_reading(reading_id)["status"] == "unexplained"
    assert store.loss_attributions(reading_id) == []


def test_multiple_detector_heads_are_reduced_to_one_ranked_row_per_site(tmp_path: Path) -> None:
    store, calculator, service = setup_services(tmp_path)
    reading_ts = datetime(2026, 8, 20, 10, 0, tzinfo=UTC)
    for detection_type, measurement, cache_key in (
        ("trash_rack_blockage", {"blockage_pct": 15}, "rack-same-site"),
        ("gate_position_mismatch", {"visual_gate_position_pct": 52}, "gate-same-site"),
    ):
        insert_event(
            store, ts=reading_ts - timedelta(minutes=5), asset_id="intake_gate",
            sensor_id="intake_gate_camera", detection_type=detection_type,
            severity="warning", confidence=0.9, measurement=measurement,
            cache_key=cache_key,
        )
    reading_id = insert_reading(store, calculator, reading_ts, actual_mw=40)

    service.attribute_reading(reading_id)

    ranked = store.loss_attributions(reading_id)
    assert len(ranked) == 1
    assert ranked[0]["asset_id"] == "intake_gate"


def test_domain_expert_rule_edit_changes_result_without_code_change(tmp_path: Path) -> None:
    store, calculator, service = setup_services(tmp_path)
    reading_ts = datetime(2026, 8, 20, 10, 0, tzinfo=UTC)
    event_confidence = 0.9
    insert_event(
        store, ts=reading_ts - timedelta(minutes=5), asset_id="penstock_valve",
        sensor_id="penstock_valve_camera", detection_type="oil_leak",
        severity="critical", confidence=event_confidence, measurement={},
        cache_key="editable-oil-map",
    )
    with store.connect() as db:
        row = db.execute(
            "SELECT params FROM attribution_rule_config WHERE rule_id = 'oil_leak'"
        ).fetchone()
        params = json.loads(row["params"])
        params["severity_map"]["critical"] = 5.0
        db.execute(
            "UPDATE attribution_rule_config SET params = ? WHERE rule_id = 'oil_leak'",
            (json.dumps(params),),
        )
    reading_id = insert_reading(store, calculator, reading_ts, actual_mw=40)

    service.attribute_reading(reading_id)
    row = store.loss_attributions(reading_id)[0]
    theoretical = store.performance_reading(reading_id)["theoretical_mw"]

    assert row["estimated_loss_mw"] == round(theoretical * 0.05 * event_confidence, 6)


def test_backfill_processes_only_triggering_unprocessed_readings(tmp_path: Path) -> None:
    store, calculator, service = setup_services(tmp_path)
    start = datetime(2026, 8, 20, 10, 0, tzinfo=UTC)
    below = insert_reading(store, calculator, start, actual_mw=52)
    above = insert_reading(store, calculator, start + timedelta(minutes=5), actual_mw=40)

    first = service.backfill()
    second = service.backfill()

    assert first == {"status": "complete", "processed": 1, "attributed": 0, "unexplained": 1}
    assert second == {"status": "complete", "processed": 0, "attributed": 0, "unexplained": 0}
    assert store.attribution_run(below) is None
    assert store.attribution_run(above)["status"] == "unexplained"
