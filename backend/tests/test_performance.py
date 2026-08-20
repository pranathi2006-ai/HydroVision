from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from backend.performance import (
    MockSourceAdapter,
    PerformanceIngestionService,
    PerformanceReading,
    PerformanceSettings,
    RealSourceAdapter,
    build_adapter,
)
from backend.reference_curves import (
    PerformanceCalculationError,
    PerformanceCalculationService,
    PerformanceCurveModel,
    import_reference_curves,
    load_reference_curve_file,
)
from backend.store import Store


UTC = timezone.utc
MOCK_CURVES = Path(__file__).resolve().parents[2] / "reference_curves" / "mock_design"


class FixedAdapter:
    def __init__(self, reading: PerformanceReading) -> None:
        self.reading = reading

    def getLatestReading(self) -> PerformanceReading:
        return self.reading


def valid_reading(ts: datetime) -> PerformanceReading:
    return PerformanceReading(
        ts=ts,
        headwater_level=118.4,
        tailwater_level=82.2,
        gate_position=61.0,
        actual_mw=48.0,
    )


def calculator(store: Store, *, nameplate_mw: float = 75) -> PerformanceCalculationService:
    if store.reference_curve_dataset() is None:
        import_reference_curves(
            store,
            MOCK_CURVES,
            dataset_name="test mock curves",
            is_demo=True,
        )
    return PerformanceCalculationService(
        store,
        PerformanceCurveModel(store, "turbine_1"),
        nameplate_capacity_mw=nameplate_mw,
    )


def test_mock_readings_are_smooth_and_physically_plausible() -> None:
    start = datetime(2026, 8, 19, 0, 0, tzinfo=UTC)
    first = MockSourceAdapter(clock=lambda: start).getLatestReading()
    second = MockSourceAdapter(clock=lambda: start + timedelta(minutes=5)).getLatestReading()

    assert 117 <= first.headwater_level <= 120
    assert 81 <= first.tailwater_level <= 84
    assert 45 <= first.gate_position <= 75
    assert 38 <= first.actual_mw <= 59
    assert abs(second.headwater_level - first.headwater_level) < 0.1
    assert abs(second.tailwater_level - first.tailwater_level) < 0.1
    assert abs(second.gate_position - first.gate_position) < 1.5
    assert abs(second.actual_mw - first.actual_mw) < 1.5


def test_adapter_selection_is_config_only() -> None:
    assert isinstance(build_adapter(PerformanceSettings(source="mock")), MockSourceAdapter)
    assert isinstance(build_adapter(PerformanceSettings(source="real")), RealSourceAdapter)


def test_effective_interval_never_outpaces_source() -> None:
    settings = PerformanceSettings(poll_interval_seconds=60, source_update_seconds=300)
    assert settings.effective_interval_seconds == 300


def test_valid_new_reading_is_stored_with_phase_two_values(tmp_path: Path) -> None:
    now = datetime(2026, 8, 19, 10, 0, tzinfo=UTC)
    store = Store(tmp_path / "hydrovision.sqlite3")
    service = PerformanceIngestionService(
        store,
        FixedAdapter(valid_reading(now)),
        PerformanceSettings(),
        calculator(store),
    )

    assert service.poll_once(now=now)
    rows = store.performance_readings_since(now - timedelta(minutes=1))
    assert len(rows) == 1
    assert rows[0]["actual_mw"] == 48.0
    assert 45 < rows[0]["theoretical_mw"] < 60
    assert rows[0]["gap_pct"] == pytest.approx((
        rows[0]["theoretical_mw"] - rows[0]["actual_mw"]
    ) / rows[0]["theoretical_mw"] * 100, abs=1e-5)


def test_new_reading_invokes_phase_four_callback_once(tmp_path: Path) -> None:
    now = datetime(2026, 8, 19, 10, 0, tzinfo=UTC)
    store = Store(tmp_path / "hydrovision.sqlite3")
    attributed: list[int] = []
    service = PerformanceIngestionService(
        store,
        FixedAdapter(valid_reading(now)),
        PerformanceSettings(),
        calculator(store),
        on_reading_stored=attributed.append,
    )

    assert service.poll_once(now=now)
    assert attributed == [store.performance_readings_since(now - timedelta(minutes=1))[0]["reading_id"]]


def test_invalid_reading_is_rejected_and_logged(tmp_path: Path, caplog) -> None:
    now = datetime(2026, 8, 19, 10, 0, tzinfo=UTC)
    store = Store(tmp_path / "hydrovision.sqlite3")
    invalid = PerformanceReading(now, 118.0, None, 140.0, 48.0)
    service = PerformanceIngestionService(
        store, FixedAdapter(invalid), PerformanceSettings(), calculator(store)
    )

    with caplog.at_level(logging.ERROR, logger="hydrovision.performance"):
        assert not service.poll_once(now=now)

    assert store.performance_readings_since(now - timedelta(days=1)) == []
    assert "performance reading rejected" in caplog.text
    assert "tailwater_level is missing" in caplog.text
    assert "gate_position=140.0 is outside" in caplog.text


def test_gap_over_twice_interval_logs_missing_window(tmp_path: Path, caplog) -> None:
    first_ts = datetime(2026, 8, 19, 9, 0, tzinfo=UTC)
    recovered_ts = first_ts + timedelta(minutes=16)
    store = Store(tmp_path / "hydrovision.sqlite3")
    assert PerformanceIngestionService(
        store,
        FixedAdapter(valid_reading(first_ts)),
        PerformanceSettings(),
        calculator(store),
    ).poll_once(now=first_ts)
    service = PerformanceIngestionService(
        store,
        FixedAdapter(valid_reading(recovered_ts)),
        PerformanceSettings(),
        calculator(store),
    )

    with caplog.at_level(logging.WARNING, logger="hydrovision.performance"):
        assert service.poll_once(now=recovered_ts)

    assert "performance reading gap detected" in caplog.text
    assert (first_ts + timedelta(minutes=5)).isoformat() in caplog.text
    assert recovered_ts.isoformat() in caplog.text


def test_stale_source_timestamp_is_rejected(tmp_path: Path, caplog) -> None:
    now = datetime(2026, 8, 19, 10, 0, tzinfo=UTC)
    store = Store(tmp_path / "hydrovision.sqlite3")
    adapter = FixedAdapter(valid_reading(now))
    service = PerformanceIngestionService(store, adapter, PerformanceSettings(), calculator(store))
    assert service.poll_once(now=now)

    with caplog.at_level(logging.ERROR, logger="hydrovision.performance"):
        assert not service.poll_once(now=now + timedelta(minutes=5))

    assert "stale or duplicated" in caplog.text
    assert len(store.performance_readings_since(now - timedelta(minutes=1))) == 1


def test_scipy_curve_interpolation_at_intermediate_points(tmp_path: Path) -> None:
    store = Store(tmp_path / "hydrovision.sqlite3")
    model = calculator(store).curves

    assert model.flow(50, 117) == 133.75
    assert model.loss(140) == 0.875
    assert model.turbine_efficiency(130, 34) == 0.885


def test_calculation_uses_required_hydropower_formula(tmp_path: Path) -> None:
    store = Store(tmp_path / "hydrovision.sqlite3")
    service = calculator(store)
    reading = PerformanceReading(
        ts=datetime(2026, 8, 19, 10, 0, tzinfo=UTC),
        headwater_level=118.0,
        tailwater_level=82.0,
        gate_position=60.0,
        actual_mw=48.0,
    )

    result = service.calculate(reading)
    expected_flow = 158.0
    expected_loss = 0.65 + (158 - 120) / (160 - 120) * (1.10 - 0.65)
    expected_net_head = 118.0 - 82.0 - expected_loss
    expected_efficiency = service.curves.turbine_efficiency(expected_flow, expected_net_head)
    expected_mw = 1000 * 9.81 * expected_flow * expected_net_head * expected_efficiency / 1_000_000

    assert result.flow_m3s == expected_flow
    assert result.loss_m == round(expected_loss, 6)
    assert result.net_head_m == round(expected_net_head, 6)
    assert result.theoretical_mw == round(expected_mw, 6)
    assert result.gap_pct == round((expected_mw - 48.0) / expected_mw * 100, 6)


def test_theoretical_output_above_nameplate_is_clamped_and_logged(tmp_path: Path, caplog) -> None:
    store = Store(tmp_path / "hydrovision.sqlite3")
    service = calculator(store, nameplate_mw=40)

    with caplog.at_level(logging.WARNING, logger="hydrovision.performance.calculation"):
        result = service.calculate(valid_reading(datetime(2026, 8, 19, 10, 0, tzinfo=UTC)))

    assert result.theoretical_mw == 40
    assert "implausible theoretical output clamped" in caplog.text


def test_curve_lookup_outside_oem_domain_rejects_new_row(tmp_path: Path, caplog) -> None:
    now = datetime(2026, 8, 19, 10, 0, tzinfo=UTC)
    store = Store(tmp_path / "hydrovision.sqlite3")
    out_of_curve_domain = PerformanceReading(now, 118.0, 82.0, 95.0, 48.0)
    service = PerformanceIngestionService(
        store,
        FixedAdapter(out_of_curve_domain),
        PerformanceSettings(gate_position_max=100),
        calculator(store),
    )

    with caplog.at_level(logging.ERROR, logger="hydrovision.performance"):
        assert not service.poll_once(now=now)

    assert store.performance_readings_since(now - timedelta(days=1)) == []
    assert "Phase 2 calculation failed" in caplog.text


def test_backfill_populates_all_historical_rows(tmp_path: Path) -> None:
    now = datetime(2026, 8, 19, 10, 0, tzinfo=UTC)
    store = Store(tmp_path / "hydrovision.sqlite3")
    with store.connect() as db:
        for offset in range(3):
            reading = valid_reading(now + timedelta(minutes=5 * offset))
            db.execute(
                """
                INSERT INTO performance_reading (
                    ts, headwater_level, tailwater_level, gate_position, actual_mw
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    reading.ts.isoformat(),
                    reading.headwater_level,
                    reading.tailwater_level,
                    reading.gate_position,
                    reading.actual_mw,
                ),
            )

    summary = calculator(store).backfill()
    rows = store.performance_readings_since(now - timedelta(minutes=1))

    assert summary == {"processed": 3, "updated": 3, "failed": 0}
    assert all(row["theoretical_mw"] is not None for row in rows)
    assert all(row["gap_pct"] is not None for row in rows)


def test_curve_model_is_cached_after_database_load(tmp_path: Path) -> None:
    store = Store(tmp_path / "hydrovision.sqlite3")
    service = calculator(store)
    with store.connect() as db:
        db.execute("DELETE FROM turbine_performance_curve")
        db.execute("DELETE FROM gate_flow_curve")
        db.execute("DELETE FROM hydraulic_loss_baseline")

    result = service.calculate(valid_reading(datetime(2026, 8, 19, 10, 0, tzinfo=UTC)))
    assert result.theoretical_mw > 0


def test_reference_curves_can_be_loaded_from_json(tmp_path: Path) -> None:
    path = tmp_path / "oem-curves.json"
    path.write_text(json.dumps({
        "turbine_performance_curve": [
            {"unit_id": "turbine_1", "flow_m3s": flow, "head_m": head, "efficiency": 0.9}
            for flow in (100, 200) for head in (30, 40)
        ],
        "gate_flow_curve": [
            {"gate_position": gate, "head_m": head, "flow_m3s": flow}
            for gate, flow in ((40, 100), (80, 200)) for head in (110, 120)
        ],
        "hydraulic_loss_baseline": [
            {"flow_m3s": 100, "loss_m": 0.5},
            {"flow_m3s": 200, "loss_m": 1.5},
        ],
    }), encoding="utf-8")

    data = load_reference_curve_file(path)

    assert len(data.turbine) == 4
    assert len(data.gate) == 4
    assert len(data.loss) == 2


def test_full_day_mock_actual_vs_theoretical_trend_is_smooth(tmp_path: Path) -> None:
    start = datetime(2026, 8, 19, 0, 0, tzinfo=UTC)
    store = Store(tmp_path / "hydrovision.sqlite3")
    service = calculator(store)
    actual: list[float] = []
    theoretical: list[float] = []
    gaps: list[float] = []
    for offset in range(0, 24 * 60 + 1, 5):
        timestamp = start + timedelta(minutes=offset)
        reading = MockSourceAdapter(clock=lambda timestamp=timestamp: timestamp).getLatestReading()
        calculated = service.calculate(reading)
        actual.append(reading.actual_mw)
        theoretical.append(calculated.theoretical_mw)
        gaps.append(calculated.gap_pct)

    assert len(theoretical) == 289
    assert all(0 < value <= 75 for value in theoretical)
    assert max(abs(right - left) for left, right in zip(actual, actual[1:])) < 1.5
    assert max(abs(right - left) for left, right in zip(theoretical, theoretical[1:])) < 1.5
    assert all(-10 < gap < 20 for gap in gaps)
