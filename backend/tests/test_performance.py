from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

from backend.performance import (
    MockSourceAdapter,
    PerformanceIngestionService,
    PerformanceReading,
    PerformanceSettings,
    RealSourceAdapter,
    build_adapter,
)
from backend.store import Store


UTC = timezone.utc


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


def test_valid_reading_is_stored_with_phase_two_columns_null(tmp_path: Path) -> None:
    now = datetime(2026, 8, 19, 10, 0, tzinfo=UTC)
    store = Store(tmp_path / "hydrovision.sqlite3")
    service = PerformanceIngestionService(
        store,
        FixedAdapter(valid_reading(now)),
        PerformanceSettings(),
    )

    assert service.poll_once(now=now)
    rows = store.performance_readings_since(now - timedelta(minutes=1))
    assert len(rows) == 1
    assert rows[0]["actual_mw"] == 48.0
    assert rows[0]["theoretical_mw"] is None
    assert rows[0]["gap_pct"] is None


def test_invalid_reading_is_rejected_and_logged(tmp_path: Path, caplog) -> None:
    now = datetime(2026, 8, 19, 10, 0, tzinfo=UTC)
    store = Store(tmp_path / "hydrovision.sqlite3")
    invalid = PerformanceReading(now, 118.0, None, 140.0, 48.0)
    service = PerformanceIngestionService(store, FixedAdapter(invalid), PerformanceSettings())

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
    ).poll_once(now=first_ts)
    service = PerformanceIngestionService(
        store,
        FixedAdapter(valid_reading(recovered_ts)),
        PerformanceSettings(),
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
    service = PerformanceIngestionService(store, adapter, PerformanceSettings())
    assert service.poll_once(now=now)

    with caplog.at_level(logging.ERROR, logger="hydrovision.performance"):
        assert not service.poll_once(now=now + timedelta(minutes=5))

    assert "stale or duplicated" in caplog.text
    assert len(store.performance_readings_since(now - timedelta(minutes=1))) == 1
