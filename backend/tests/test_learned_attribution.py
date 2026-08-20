from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from backend.learned_attribution import (
    LearnedAttributionService,
    LearnedAttributionSettings,
    LearnedAttributionTrainer,
)
from backend.store import Store


UTC = timezone.utc


def settings() -> LearnedAttributionSettings:
    return LearnedAttributionSettings(
        minimum_training_rows=20,
        minimum_rows_per_defect=5,
        validation_fraction=0.25,
        minimum_shadow_feedback=10,
        minimum_shadow_days=0,
        promotion_brier_margin=0.02,
        retrain_days=30,
        retrain_new_feedback=10,
        scheduler_check_seconds=60,
    )


def insert_example(
    store: Store,
    index: int,
    *,
    confirmed: bool,
    detection_type: str = "oil_leak",
    asset_id: str = "penstock_valve",
    shadow_model_id: str | None = None,
    shadow_probability: float | None = None,
) -> tuple[int, int, int]:
    reading_ts = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(hours=index)
    severity = "critical" if confirmed else "observation"
    gap_pct = 15.0 if confirmed else 5.5
    rule_estimate = 2.5 if confirmed else 0.3
    sensor_id = "main_transformer_thermal" if asset_id == "main_transformer" else "penstock_valve_camera"
    measurement = {"delta_t_c": 18.0 if confirmed else 2.0} if detection_type == "thermal_hotspot" else {
        "affected_area_pct": 14.0 if confirmed else 1.0,
    }
    with store.connect() as db:
        reading = db.execute(
            """
            INSERT INTO performance_reading (
                ts, headwater_level, tailwater_level, gate_position,
                theoretical_mw, actual_mw, gap_pct
            ) VALUES (?, 118, 82, 60, 50, ?, ?)
            """,
            (reading_ts.isoformat(), 50 * (1 - gap_pct / 100), gap_pct),
        )
        reading_id = int(reading.lastrowid)
        event = db.execute(
            """
            INSERT INTO detection_event (
                ts, asset_id, sensor_id, detection_type, defect_present,
                severity, confidence, measurement, engine, cache_key, created_at
            ) VALUES (?, ?, ?, ?, 1, ?, ?, ?, 'test', ?, ?)
            """,
            (
                (reading_ts - timedelta(minutes=10)).isoformat(), asset_id, sensor_id,
                detection_type, severity, 0.75, json.dumps(measurement),
                f"phase6-{index}-{detection_type}", reading_ts.isoformat(),
            ),
        )
        event_id = int(event.lastrowid)
        attribution = db.execute(
            """
            INSERT INTO loss_attribution (
                reading_id, asset_id, event_id, estimated_loss_mw, confidence,
                method, rule_estimate_mw, rule_confidence, shadow_estimate_mw,
                shadow_probability, shadow_model_id
            ) VALUES (?, ?, ?, ?, 0.75, 'rule_based', ?, 0.75, ?, ?, ?)
            """,
            (
                reading_id, asset_id, event_id, rule_estimate, rule_estimate,
                rule_estimate * shadow_probability if shadow_probability is not None else None,
                shadow_probability, shadow_model_id,
            ),
        )
        attribution_id = int(attribution.lastrowid)
    store.add_attribution_feedback(
        attribution_id,
        confirmed=confirmed,
        notes="inspection verified outcome",
        confirmed_by="plant.engineer",
    )
    return reading_id, event_id, attribution_id


def train_shadow(store: Store) -> dict:
    for index in range(40):
        insert_example(store, index, confirmed=index % 2 == 0)
    return LearnedAttributionTrainer(store, settings()).train()


def test_feedback_is_unique_and_training_always_starts_in_shadow(tmp_path: Path) -> None:
    store = Store(tmp_path / "hydrovision.sqlite3")
    model = train_shadow(store)

    assert store.confirmed_attribution_count() == 40
    assert model["status"] == "shadow"
    assert model["purpose"] == "loss_attribution"
    assert model["model_type"] == "logistic_regression"
    assert model["metrics"]["validation"]["precision"] >= 0.8
    assert model["metrics"]["validation"]["recall"] >= 0.8
    assert store.loss_model("active") is None
    first_attribution = store.labeled_attribution_rows()[0]["attribution_id"]
    with pytest.raises(ValueError, match="already exists"):
        store.add_attribution_feedback(
            first_attribution, confirmed=True, notes=None, confirmed_by="second.engineer",
        )


def test_shadow_scores_separately_and_rare_defect_stays_rule_based(tmp_path: Path) -> None:
    store = Store(tmp_path / "hydrovision.sqlite3")
    model = train_shadow(store)
    service = LearnedAttributionService(store, settings())

    reading_id, event_id, _ = insert_example(store, 100, confirmed=True)
    scored = service.score_contributions(reading_id, [{
        "asset_id": "penstock_valve",
        "event_id": event_id,
        "estimated_loss_mw": 2.5,
        "confidence": 0.75,
    }])[0]
    assert scored["method"] == "rule_based"
    assert scored["estimated_loss_mw"] == 2.5
    assert scored["shadow_model_id"] == model["model_id"]
    assert 0 < scored["shadow_probability"] < 1
    assert scored["shadow_explanation"]["top_feature_contributions"]

    thermal_reading, thermal_event, _ = insert_example(
        store, 101, confirmed=True, detection_type="thermal_hotspot", asset_id="main_transformer",
    )
    rare = service.score_contributions(thermal_reading, [{
        "asset_id": "main_transformer",
        "event_id": thermal_event,
        "estimated_loss_mw": 1.2,
        "confidence": 0.75,
    }])[0]
    assert rare["method"] == "rule_based"
    assert "shadow_model_id" not in rare


def test_promotion_requires_real_comparison_clear_win_and_explicit_human_confirmation(tmp_path: Path) -> None:
    store = Store(tmp_path / "hydrovision.sqlite3")
    model = train_shadow(store)
    service = LearnedAttributionService(store, settings())
    for index in range(200, 212):
        confirmed = index % 2 == 0
        insert_example(
            store,
            index,
            confirmed=confirmed,
            shadow_model_id=model["model_id"],
            shadow_probability=0.95 if confirmed else 0.05,
        )

    comparison = service.compare_shadow(model["model_id"])
    assert comparison["clearly_better"] is True
    assert comparison["shadow"]["precision"] > comparison["rule_baseline"]["precision"]

    with pytest.raises(PermissionError, match="explicit confirmation"):
        service.promote(
            model["model_id"], approved_by="chief.engineer", confirmation="yes",
        )
    with store.connect() as db, pytest.raises(sqlite3.IntegrityError, match="explicit approval"):
        db.execute(
            "UPDATE correlation_model_version SET status = 'active' WHERE model_id = ?",
            (model["model_id"],),
        )

    promoted = service.promote(
        model["model_id"],
        approved_by="chief.engineer",
        confirmation=f"PROMOTE {model['model_id']}",
        approval_notes="Shadow review approved in the monthly reliability meeting.",
    )
    assert promoted["status"] == "active"
    assert promoted["approved_by"] == "chief.engineer"
    active_reading, active_event, _ = insert_example(store, 300, confirmed=True)
    active_score = service.score_contributions(active_reading, [{
        "asset_id": "penstock_valve",
        "event_id": active_event,
        "estimated_loss_mw": 2.5,
        "confidence": 0.75,
    }])[0]
    assert active_score["method"] == "learned"
    assert active_score["model_id"] == model["model_id"]
    assert 0 < active_score["estimated_loss_mw"] <= active_score["rule_estimate_mw"]


def test_scheduler_entry_point_never_promotes(tmp_path: Path) -> None:
    store = Store(tmp_path / "hydrovision.sqlite3")
    for index in range(40):
        insert_example(store, index, confirmed=index % 2 == 0)
    result = LearnedAttributionTrainer(store, settings()).train_if_due()

    assert result["status"] == "trained_shadow"
    assert store.loss_model("shadow") is not None
    assert store.loss_model("active") is None
