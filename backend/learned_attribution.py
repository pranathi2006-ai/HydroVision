from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .store import Store, utc_now


logger = logging.getLogger("hydrovision.performance.learned_attribution")
SEVERITY_SCORE = {"normal": 0.0, "observation": 0.33, "warning": 0.66, "critical": 1.0}
MEASUREMENT_KEYS = (
    "blockage_pct",
    "delta_t_c",
    "mismatch_pct_points",
    "visual_gate_position_pct",
    "pitting_area_pct",
    "affected_area_pct",
)


@dataclass(frozen=True)
class LearnedAttributionSettings:
    minimum_training_rows: int = 20
    minimum_rows_per_defect: int = 10
    validation_fraction: float = 0.2
    minimum_shadow_feedback: int = 30
    minimum_shadow_days: int = 14
    promotion_brier_margin: float = 0.02
    retrain_days: int = 30
    retrain_new_feedback: int = 25
    scheduler_check_seconds: int = 24 * 60 * 60

    @classmethod
    def from_env(cls) -> "LearnedAttributionSettings":
        settings = cls(
            minimum_training_rows=int(os.getenv("HYDROVISION_LEARNED_MIN_TRAINING_ROWS", "20")),
            minimum_rows_per_defect=int(os.getenv("HYDROVISION_LEARNED_MIN_DEFECT_ROWS", "10")),
            validation_fraction=float(os.getenv("HYDROVISION_LEARNED_VALIDATION_FRACTION", "0.2")),
            minimum_shadow_feedback=int(os.getenv("HYDROVISION_LEARNED_MIN_SHADOW_FEEDBACK", "30")),
            minimum_shadow_days=int(os.getenv("HYDROVISION_LEARNED_MIN_SHADOW_DAYS", "14")),
            promotion_brier_margin=float(os.getenv("HYDROVISION_LEARNED_PROMOTION_BRIER_MARGIN", "0.02")),
            retrain_days=int(os.getenv("HYDROVISION_LEARNED_RETRAIN_DAYS", "30")),
            retrain_new_feedback=int(os.getenv("HYDROVISION_LEARNED_RETRAIN_NEW_FEEDBACK", "25")),
            scheduler_check_seconds=int(os.getenv("HYDROVISION_LEARNED_SCHEDULER_CHECK_SECONDS", "86400")),
        )
        if settings.minimum_training_rows < 4 or settings.minimum_rows_per_defect < 1:
            raise ValueError("learned-attribution sample limits are invalid")
        if not 0.1 <= settings.validation_fraction <= 0.4:
            raise ValueError("learned-attribution validation fraction must be between 0.1 and 0.4")
        if (
            settings.minimum_shadow_feedback < 1
            or settings.minimum_shadow_days < 0
            or settings.promotion_brier_margin < 0
        ):
            raise ValueError("learned-attribution promotion settings are invalid")
        if settings.retrain_days < 1 or settings.retrain_new_feedback < 1:
            raise ValueError("learned-attribution retraining settings are invalid")
        if settings.scheduler_check_seconds < 60:
            raise ValueError("learned-attribution scheduler must not run more than once per minute")
        return settings


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value).astimezone(timezone.utc)


def build_features(row: dict) -> dict[str, float | str]:
    measurement = row.get("measurement") or {}
    measurement_key = "none"
    measurement_value = 0.0
    for key in MEASUREMENT_KEYS:
        value = measurement.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            measurement_key = key
            measurement_value = float(value)
            break

    reading_ts = _parse_timestamp(str(row["reading_ts"]))
    maintenance_at = row.get("last_maintenance_at")
    if maintenance_at:
        maintenance_age_days = max(
            0.0, (reading_ts - _parse_timestamp(str(maintenance_at))).total_seconds() / 86400,
        )
        maintenance_known = 1.0
    else:
        # Unknown maintenance history is represented explicitly rather than
        # pretending a live maintenance feed exists.
        maintenance_age_days = 365.0
        maintenance_known = 0.0

    return {
        "visual_severity_score": SEVERITY_SCORE.get(str(row.get("severity")), 0.0),
        "measurement_value": measurement_value,
        "measurement_type": measurement_key,
        "rule_estimate_mw": float(row["rule_estimate_mw"]),
        "asset": str(row["asset_id"]),
        "defect_type": str(row["detection_type"]),
        "asset_criticality": float(row.get("criticality") or 0.5),
        "maintenance_age_days": maintenance_age_days,
        "maintenance_known": maintenance_known,
        "gap_pct": float(row.get("gap_pct") or 0.0),
    }


def _classification_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    predictions = probabilities >= 0.5
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0,
    )
    return {
        "precision": round(float(precision), 6),
        "recall": round(float(recall), 6),
        "f1": round(float(f1), 6),
        "brier_score": round(float(brier_score_loss(labels, probabilities)), 6),
    }


class StoredLogisticModel:
    def __init__(self, record: dict, minimum_rows_per_defect: int) -> None:
        self.record = record
        self.model_id = str(record["model_id"])
        self.artifact = record["artifact"]
        self.defect_counts = record["defect_counts"] or {}
        self.minimum_rows_per_defect = minimum_rows_per_defect

    def supports(self, defect_type: str) -> bool:
        return int(self.defect_counts.get(defect_type, 0)) >= self.minimum_rows_per_defect

    def predict(self, features: dict[str, float | str]) -> tuple[float, dict]:
        names = self.artifact["feature_names"]
        vector = []
        for name in names:
            if "=" in name:
                key, value = name.split("=", 1)
                vector.append(1.0 if str(features.get(key)) == value else 0.0)
            else:
                raw = features.get(name, 0.0)
                vector.append(float(raw) if isinstance(raw, (int, float)) else 0.0)
        mean = np.asarray(self.artifact["scaler_mean"], dtype=float)
        scale = np.asarray(self.artifact["scaler_scale"], dtype=float)
        standardized = (np.asarray(vector, dtype=float) - mean) / np.where(scale == 0, 1.0, scale)
        coefficients = np.asarray(self.artifact["coefficients"], dtype=float)
        contributions = standardized * coefficients
        logit = float(np.dot(standardized, coefficients) + float(self.artifact["intercept"]))
        probability = 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, logit))))
        ranked = sorted(
            (
                {"feature": name, "contribution": round(float(value), 6)}
                for name, value in zip(names, contributions)
            ),
            key=lambda item: abs(item["contribution"]),
            reverse=True,
        )[:5]
        return probability, {
            "model_id": self.model_id,
            "probability_confirmed": round(probability, 6),
            "top_feature_contributions": ranked,
        }


class LearnedAttributionTrainer:
    def __init__(self, store: Store, settings: LearnedAttributionSettings) -> None:
        self.store = store
        self.settings = settings

    def train(self) -> dict:
        rows = self.store.labeled_attribution_rows()
        if len(rows) < self.settings.minimum_training_rows:
            raise ValueError(
                f"need at least {self.settings.minimum_training_rows} confirmed outcomes; found {len(rows)}"
            )
        labels = np.asarray([int(row["confirmed"]) for row in rows], dtype=int)
        counts = Counter(labels.tolist())
        if len(counts) != 2 or min(counts.values()) < 2:
            raise ValueError("confirmed outcomes must include at least two positive and two negative labels")

        features = [build_features(row) for row in rows]
        train_features, validation_features, train_labels, validation_labels, _, validation_rows = train_test_split(
            features,
            labels,
            rows,
            test_size=self.settings.validation_fraction,
            random_state=42,
            stratify=labels,
        )
        vectorizer = DictVectorizer(sparse=False)
        train_matrix = vectorizer.fit_transform(train_features)
        validation_matrix = vectorizer.transform(validation_features)
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_matrix)
        validation_scaled = scaler.transform(validation_matrix)
        classifier = LogisticRegression(
            class_weight="balanced", max_iter=500, random_state=42,
        )
        classifier.fit(train_scaled, train_labels)
        learned_probabilities = classifier.predict_proba(validation_scaled)[:, 1]
        rule_probabilities = np.asarray(
            [min(1.0, max(0.0, float(row["rule_confidence"]))) for row in validation_rows],
            dtype=float,
        )

        artifact = {
            "feature_names": vectorizer.get_feature_names_out().tolist(),
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "coefficients": classifier.coef_[0].tolist(),
            "intercept": float(classifier.intercept_[0]),
            "threshold": 0.5,
        }
        feature_importance = sorted(
            (
                {"feature": name, "absolute_coefficient": round(abs(float(value)), 6)}
                for name, value in zip(artifact["feature_names"], artifact["coefficients"])
            ),
            key=lambda item: item["absolute_coefficient"],
            reverse=True,
        )
        metrics = {
            "validation": _classification_metrics(validation_labels, learned_probabilities),
            "rule_baseline": _classification_metrics(validation_labels, rule_probabilities),
            "feature_importance": feature_importance,
            "positive_training_rows": int(counts[1]),
            "negative_training_rows": int(counts[0]),
        }
        now = utc_now()
        digest = hashlib.sha256(
            json.dumps({"artifact": artifact, "feedback_ids": [row["feedback_id"] for row in rows]}, sort_keys=True).encode()
        ).hexdigest()[:10]
        model_id = f"lossattr-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{digest}"
        active = self.store.loss_model("active")
        model = {
            "model_id": model_id,
            "created_at": now,
            "trained_at": now,
            "training_rows": len(train_features),
            "validation_rows": len(validation_features),
            "metrics": metrics,
            "artifact": artifact,
            "defect_counts": dict(Counter(row["detection_type"] for row in rows)),
            "shadow_started_at": now,
            "supersedes_model_id": active["model_id"] if active else None,
        }
        self.store.save_loss_model_version(model)
        logger.info(
            "loss-attribution model trained model_id=%s status=shadow training_rows=%s validation_rows=%s",
            model_id, len(train_features), len(validation_features),
        )
        return self.store.loss_model_by_id(model_id) or model

    def train_if_due(self) -> dict:
        feedback_count = self.store.confirmed_attribution_count()
        if feedback_count < self.settings.minimum_training_rows:
            return {"status": "insufficient_feedback", "feedback_count": feedback_count}
        candidates = [model for model in (self.store.loss_model("shadow"), self.store.loss_model("active")) if model]
        latest = max(candidates, key=lambda item: item.get("trained_at") or "") if candidates else None
        if latest:
            rows_at_training = int(latest.get("training_rows") or 0) + int(latest.get("validation_rows") or 0)
            new_feedback = feedback_count - rows_at_training
            age_days = (datetime.now(timezone.utc) - _parse_timestamp(latest["trained_at"])).days
            if new_feedback <= 0 or (
                new_feedback < self.settings.retrain_new_feedback and age_days < self.settings.retrain_days
            ):
                return {"status": "not_due", "feedback_count": feedback_count, "new_feedback": new_feedback}
        model = self.train()
        return {"status": "trained_shadow", "model_id": model["model_id"]}


class LearnedAttributionService:
    def __init__(self, store: Store, settings: LearnedAttributionSettings) -> None:
        self.store = store
        self.settings = settings

    def score_contributions(self, reading_id: int, contributions: list[dict]) -> list[dict]:
        active_record = self.store.loss_model("active")
        shadow_record = self.store.loss_model("shadow")
        active = StoredLogisticModel(active_record, self.settings.minimum_rows_per_defect) if active_record else None
        shadow = StoredLogisticModel(shadow_record, self.settings.minimum_rows_per_defect) if shadow_record else None
        scored = []
        for original in contributions:
            contribution = dict(original)
            rule_estimate = float(contribution["estimated_loss_mw"])
            rule_confidence = float(contribution["confidence"])
            contribution["rule_estimate_mw"] = rule_estimate
            contribution["rule_confidence"] = rule_confidence
            context = self.store.attribution_feature_row(
                reading_id,
                int(contribution["event_id"]),
                str(contribution["asset_id"]),
                rule_estimate,
                rule_confidence,
            )
            features = build_features(context)
            defect_type = str(context["detection_type"])
            if active and active.supports(defect_type):
                probability, explanation = active.predict(features)
                contribution.update({
                    "estimated_loss_mw": round(rule_estimate * probability, 6),
                    "confidence": round(probability, 4),
                    "method": "learned",
                    "model_id": active.model_id,
                    "model_explanation": explanation,
                })
            else:
                contribution.update({"method": "rule_based", "model_id": None})
            if shadow and shadow.supports(defect_type):
                probability, explanation = shadow.predict(features)
                contribution.update({
                    "shadow_estimate_mw": round(rule_estimate * probability, 6),
                    "shadow_probability": round(probability, 6),
                    "shadow_model_id": shadow.model_id,
                    "shadow_explanation": explanation,
                })
            scored.append(contribution)
        return scored

    def compare_shadow(self, model_id: str) -> dict:
        model = self.store.loss_model_by_id(model_id)
        if model is None or model["status"] != "shadow":
            raise ValueError("comparison requires a current shadow loss-attribution model")
        rows = self.store.shadow_feedback_rows(model_id)
        shadow_age_days = (
            datetime.now(timezone.utc) - _parse_timestamp(model["shadow_started_at"])
        ).days
        if shadow_age_days < self.settings.minimum_shadow_days:
            raise ValueError(
                f"shadow period must run at least {self.settings.minimum_shadow_days} days; "
                f"current age is {shadow_age_days} days"
            )
        if len(rows) < self.settings.minimum_shadow_feedback:
            raise ValueError(
                f"need at least {self.settings.minimum_shadow_feedback} confirmed shadow outcomes; found {len(rows)}"
            )
        labels = np.asarray([int(row["confirmed"]) for row in rows], dtype=int)
        shadow_probabilities = np.asarray([row["shadow_probability"] for row in rows], dtype=float)
        rule_probabilities = np.asarray([row["rule_probability"] for row in rows], dtype=float)
        shadow_metrics = _classification_metrics(labels, shadow_probabilities)
        rule_metrics = _classification_metrics(labels, rule_probabilities)
        clearly_better = (
            shadow_metrics["brier_score"] + self.settings.promotion_brier_margin
            < rule_metrics["brier_score"]
            and shadow_metrics["precision"] >= rule_metrics["precision"]
            and shadow_metrics["recall"] >= 0.5
        )
        comparison = {
            "evaluation_rows": len(rows),
            "shadow_age_days": shadow_age_days,
            "minimum_shadow_days": self.settings.minimum_shadow_days,
            "shadow": shadow_metrics,
            "rule_baseline": rule_metrics,
            "required_brier_margin": self.settings.promotion_brier_margin,
            "clearly_better": clearly_better,
            "evaluated_at": utc_now(),
        }
        self.store.save_model_comparison(model_id, comparison)
        return comparison

    def promote(
        self,
        model_id: str,
        *,
        approved_by: str,
        confirmation: str,
        approval_notes: str | None = None,
    ) -> dict:
        reviewer = approved_by.strip()
        if not reviewer:
            raise ValueError("approved_by is required")
        if confirmation != f"PROMOTE {model_id}":
            raise PermissionError(f"explicit confirmation must equal 'PROMOTE {model_id}'")
        model = self.store.loss_model_by_id(model_id)
        if model is None or model["status"] != "shadow":
            raise ValueError("only a shadow loss-attribution model can be promoted")
        comparison = model.get("comparison_metrics")
        if not comparison or not comparison.get("clearly_better"):
            raise ValueError("shadow model has not demonstrated a clear win over the rule baseline")
        self.store.activate_loss_model(
            model_id, approved_by=reviewer, approval_notes=approval_notes,
        )
        logger.warning(
            "loss-attribution model promoted model_id=%s approved_by=%s",
            model_id, reviewer,
        )
        return self.store.loss_model_by_id(model_id) or {"model_id": model_id, "status": "active"}


class LearnedAttributionRetrainingScheduler:
    """Periodic shadow-only retraining. This class has no promotion path."""

    def __init__(self, trainer: LearnedAttributionTrainer, settings: LearnedAttributionSettings) -> None:
        self.trainer = trainer
        self.settings = settings
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run(), name="loss-attribution-retraining")

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None

    async def _run(self) -> None:
        while True:
            try:
                result = await asyncio.to_thread(self.trainer.train_if_due)
                if result["status"] == "trained_shadow":
                    logger.info("scheduled loss-attribution retraining created %s", result["model_id"])
            except Exception:
                logger.exception("scheduled loss-attribution retraining failed; rule fallback remains active")
            await asyncio.sleep(self.settings.scheduler_check_seconds)
