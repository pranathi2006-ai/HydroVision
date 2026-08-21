from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from backend.store import Store
from backend.verification import AttributionVerificationService, VerificationSettings


UTC = timezone.utc


def settings(**overrides) -> VerificationSettings:
    values = {
        "wait_days": 7,
        "comparison_window_days": 30,
        "retry_days": 3,
        "operating_tolerance_pct": 0.05,
        "minimum_samples": 20,
        "significance_alpha": 0.05,
        "minimum_effect_gap_pct": 0.5,
        "null_alert_threshold": 0.5,
        "scheduler_check_seconds": 60,
    }
    values.update(overrides)
    return VerificationSettings(**values)


def seed_attribution(store: Store, index: int = 1) -> tuple[int, int]:
    ts = datetime(2026, 4, 1, tzinfo=UTC)
    with store.connect() as db:
        reading_id = db.execute(
            """
            INSERT INTO performance_reading (
                ts, headwater_level, tailwater_level, gate_position,
                theoretical_mw, actual_mw, gap_pct
            ) VALUES (?, 118, 82, 60, 50, 44, 12)
            """,
            (ts.isoformat(),),
        ).lastrowid
        event_id = db.execute(
            """
            INSERT INTO detection_event (
                ts, asset_id, sensor_id, detection_type, defect_present,
                severity, confidence, measurement, engine, cache_key, created_at
            ) VALUES (?, 'penstock_valve', 'penstock_valve_camera', 'oil_leak',
                      1, 'critical', 0.9, ?, 'test', ?, ?)
            """,
            (
                ts.isoformat(), json.dumps({"affected_area_pct": 12.0}),
                f"verification-event-{index}", ts.isoformat(),
            ),
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
    return int(event_id), int(attribution_id)


def add_reading(store: Store, ts: datetime, gap: float, index: int) -> None:
    with store.connect() as db:
        db.execute(
            """
            INSERT INTO performance_reading (
                ts, headwater_level, tailwater_level, gate_position,
                theoretical_mw, actual_mw, gap_pct
            ) VALUES (?, ?, 82, ?, 50, ?, ?)
            """,
            (
                ts.isoformat(), 118 + (index % 3) * 0.1,
                60 + (index % 4) * 0.1, 50 * (1 - gap / 100), gap,
            ),
        )


def create_closed_order(
    store: Store,
    attribution_id: int,
    *,
    opened: datetime,
    closed: datetime,
    asset_id: str = "penstock_valve",
) -> int:
    with store.connect() as db:
        order_id = db.execute(
            """
            INSERT INTO work_order (
                asset_id, attribution_id, status, opened_at,
                dispatch_approved_by, dispatch_approved_at
            ) VALUES (?, ?, 'approved', ?, 'maintenance.lead', ?)
            """,
            (asset_id, attribution_id, opened.isoformat(), (opened + timedelta(hours=1)).isoformat()),
        ).lastrowid
        db.execute(
            "UPDATE work_order SET status = 'closed', closed_at = ? WHERE work_order_id = ?",
            (closed.isoformat(), order_id),
        )
    return int(order_id)


def seed_matched_windows(
    store: Store,
    opened: datetime,
    closed: datetime,
    *,
    before_gap: float,
    after_gap: float,
    count: int = 24,
) -> None:
    for index in range(count):
        add_reading(store, opened - timedelta(days=10, hours=index), before_gap + index * 0.01, index)
        add_reading(store, closed + timedelta(hours=index + 1), after_gap + index * 0.01, index)


def test_closed_approved_order_auto_confirms_from_matched_conditions(tmp_path: Path) -> None:
    store = Store(tmp_path / "hydrovision.sqlite3")
    _, attribution_id = seed_attribution(store)
    opened = datetime(2026, 5, 1, tzinfo=UTC)
    closed = datetime(2026, 5, 3, tzinfo=UTC)
    create_closed_order(store, attribution_id, opened=opened, closed=closed)
    seed_matched_windows(store, opened, closed, before_gap=12.0, after_gap=4.0)
    service = AttributionVerificationService(store, settings())

    summary = service.run_due(closed + timedelta(days=15))

    assert summary["confirmed"] == 1
    with store.connect() as db:
        feedback = db.execute(
            "SELECT * FROM attribution_feedback WHERE attribution_id = ?", (attribution_id,),
        ).fetchone()
    assert feedback["confirmed"] == 1
    assert feedback["confirmed_by"] == "system_auto"
    assert feedback["verification_method"] == "auto_matched_condition"
    assert feedback["sample_size_before"] >= 20
    assert feedback["gap_before_mean"] > feedback["gap_after_mean"]
    assert feedback["p_value"] < 0.05


def test_insufficient_samples_stay_null_and_raise_monitoring_alert(tmp_path: Path) -> None:
    store = Store(tmp_path / "hydrovision.sqlite3")
    service = AttributionVerificationService(store, settings(null_alert_threshold=0.1))
    for index in range(5):
        _, attribution_id = seed_attribution(store, index + 10)
        opened = datetime(2026, 5, 1 + index, tzinfo=UTC)
        closed = opened + timedelta(days=1)
        create_closed_order(store, attribution_id, opened=opened, closed=closed)
        seed_matched_windows(store, opened, closed, before_gap=10, after_gap=8, count=3)
    service.run_due(datetime(2026, 7, 20, tzinfo=UTC))

    monitoring = service.monitoring()

    assert monitoring["inconclusive_null"] == 5
    assert monitoring["inconclusive_null_pct"] == 100.0
    assert monitoring["alert"] is not None


def test_significant_post_fix_worsening_auto_rejects_attribution(tmp_path: Path) -> None:
    store = Store(tmp_path / "hydrovision.sqlite3")
    _, attribution_id = seed_attribution(store, 35)
    opened = datetime(2026, 5, 1, tzinfo=UTC)
    closed = datetime(2026, 5, 3, tzinfo=UTC)
    create_closed_order(store, attribution_id, opened=opened, closed=closed)
    seed_matched_windows(store, opened, closed, before_gap=4.0, after_gap=11.0)

    summary = AttributionVerificationService(store, settings()).run_due(
        closed + timedelta(days=15)
    )

    assert summary["rejected"] == 1
    with store.connect() as db:
        feedback = db.execute(
            "SELECT confirmed, p_value FROM attribution_feedback WHERE attribution_id = ?",
            (attribution_id,),
        ).fetchone()
    assert feedback["confirmed"] == 0
    assert feedback["p_value"] < 0.05


def test_concurrent_fix_blocks_both_orders(tmp_path: Path) -> None:
    store = Store(tmp_path / "hydrovision.sqlite3")
    _, first = seed_attribution(store, 40)
    _, second = seed_attribution(store, 41)
    opened = datetime(2026, 5, 1, tzinfo=UTC)
    closed = datetime(2026, 5, 3, tzinfo=UTC)
    create_closed_order(store, first, opened=opened, closed=closed)
    create_closed_order(store, second, opened=opened + timedelta(days=1), closed=closed + timedelta(days=2))
    seed_matched_windows(store, opened, closed, before_gap=12, after_gap=4)
    service = AttributionVerificationService(store, settings())

    summary = service.run_due(closed + timedelta(days=20))

    assert summary["inconclusive"] == 2
    with store.connect() as db:
        rows = db.execute(
            "SELECT confirmed, notes FROM attribution_feedback ORDER BY feedback_id"
        ).fetchall()
    assert all(row["confirmed"] is None for row in rows)
    assert all("another work order" in row["notes"] for row in rows)


def test_dispatch_and_closure_still_require_manual_approval(tmp_path: Path) -> None:
    store = Store(tmp_path / "hydrovision.sqlite3")
    _, attribution_id = seed_attribution(store, 50)
    with store.connect() as db:
        order_id = db.execute(
            """
            INSERT INTO work_order (asset_id, attribution_id, status, opened_at)
            VALUES ('penstock_valve', ?, 'pending_approval', ?)
            """,
            (attribution_id, datetime(2026, 5, 1, tzinfo=UTC).isoformat()),
        ).lastrowid
    with pytest.raises(sqlite3.IntegrityError, match="manual dispatch approval"):
        with store.connect() as db:
            db.execute(
                "UPDATE work_order SET status = 'dispatched' WHERE work_order_id = ?",
                (order_id,),
            )
    with store.connect() as db:
        status = db.execute(
            "SELECT status FROM work_order WHERE work_order_id = ?", (order_id,),
        ).fetchone()["status"]
    assert status == "pending_approval"


def test_manual_history_backtest_reproduces_direction(tmp_path: Path) -> None:
    store = Store(tmp_path / "hydrovision.sqlite3")
    _, attribution_id = seed_attribution(store, 60)
    opened = datetime(2026, 5, 1, tzinfo=UTC)
    closed = datetime(2026, 5, 3, tzinfo=UTC)
    create_closed_order(store, attribution_id, opened=opened, closed=closed)
    seed_matched_windows(store, opened, closed, before_gap=11, after_gap=4)
    store.add_attribution_feedback(
        attribution_id, confirmed=True, notes="Engineer confirmed recovery", confirmed_by="engineer",
    )
    service = AttributionVerificationService(store, settings())

    backtest = service.backtest_manual_history()

    assert backtest["comparable_rows"] == 1
    assert backtest["agreement_rate"] == 1.0
    assert backtest["majority_reproduced"] is True
