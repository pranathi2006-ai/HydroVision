from __future__ import annotations

import asyncio
import logging
import math
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
from scipy.stats import ttest_rel

from .store import Store


logger = logging.getLogger("hydrovision.attribution.verification")


@dataclass(frozen=True)
class VerificationSettings:
    wait_days: int = 14
    comparison_window_days: int = 45
    retry_days: int = 7
    operating_tolerance_pct: float = 0.05
    minimum_samples: int = 20
    significance_alpha: float = 0.05
    minimum_effect_gap_pct: float = 0.5
    null_alert_threshold: float = 0.5
    scheduler_check_seconds: int = 60 * 60

    @classmethod
    def from_env(cls) -> "VerificationSettings":
        settings = cls(
            wait_days=int(os.getenv("HYDROVISION_VERIFICATION_WAIT_DAYS", "14")),
            comparison_window_days=int(os.getenv("HYDROVISION_VERIFICATION_WINDOW_DAYS", "45")),
            retry_days=int(os.getenv("HYDROVISION_VERIFICATION_RETRY_DAYS", "7")),
            operating_tolerance_pct=float(os.getenv("HYDROVISION_VERIFICATION_TOLERANCE_PCT", "5")) / 100,
            minimum_samples=int(os.getenv("HYDROVISION_VERIFICATION_MIN_SAMPLES", "20")),
            significance_alpha=float(os.getenv("HYDROVISION_VERIFICATION_ALPHA", "0.05")),
            minimum_effect_gap_pct=float(os.getenv("HYDROVISION_VERIFICATION_MIN_EFFECT_GAP_PCT", "0.5")),
            null_alert_threshold=float(os.getenv("HYDROVISION_VERIFICATION_NULL_ALERT_PCT", "50")) / 100,
            scheduler_check_seconds=int(os.getenv("HYDROVISION_VERIFICATION_CHECK_SECONDS", "3600")),
        )
        if not 7 <= settings.wait_days <= 60:
            raise ValueError("verification wait must be between 7 and 60 days")
        if settings.comparison_window_days < settings.wait_days:
            raise ValueError("verification window must be at least as long as its wait")
        if settings.retry_days < 1 or settings.minimum_samples < 2:
            raise ValueError("verification retry/sample settings are invalid")
        if not 0 < settings.operating_tolerance_pct <= 0.25:
            raise ValueError("verification operating tolerance must be between 0 and 25 percent")
        if not 0 < settings.significance_alpha < 0.5 or settings.minimum_effect_gap_pct < 0:
            raise ValueError("verification statistical settings are invalid")
        if not 0 <= settings.null_alert_threshold <= 1:
            raise ValueError("verification NULL alert threshold is invalid")
        if settings.scheduler_check_seconds < 60:
            raise ValueError("verification scheduler must not run more than once per minute")
        return settings


def _utc(value: str) -> datetime:
    return datetime.fromisoformat(value).astimezone(timezone.utc)


def match_operating_conditions(
    before: list[dict], after: list[dict], tolerance: float,
) -> tuple[list[float], list[float]]:
    """Greedy one-to-one nearest matching within headwater/gate tolerance."""
    unused = set(range(len(before)))
    matched_before: list[float] = []
    matched_after: list[float] = []
    for after_row in after:
        gate_after = float(after_row["gate_position"])
        head_after = float(after_row["headwater_level"])
        candidates: list[tuple[float, int]] = []
        for index in unused:
            before_row = before[index]
            gate_before = float(before_row["gate_position"])
            head_before = float(before_row["headwater_level"])
            gate_delta = abs(gate_before - gate_after) / max(abs(gate_after), 1.0)
            head_delta = abs(head_before - head_after) / max(abs(head_after), 1.0)
            if gate_delta <= tolerance and head_delta <= tolerance:
                candidates.append((gate_delta + head_delta, index))
        if not candidates:
            continue
        _, selected = min(candidates)
        unused.remove(selected)
        matched_before.append(float(before[selected]["gap_pct"]))
        matched_after.append(float(after_row["gap_pct"]))
    return matched_before, matched_after


class AttributionVerificationService:
    def __init__(self, store: Store, settings: VerificationSettings) -> None:
        self.store = store
        self.settings = settings

    def evaluate(
        self,
        *,
        work_order_id: int,
        asset_id: str,
        opened_at: str,
        closed_at: str,
        now: datetime,
        check_concurrent: bool = True,
    ) -> dict:
        opened = _utc(opened_at)
        closed = _utc(closed_at)
        before_start = opened - timedelta(days=self.settings.comparison_window_days)
        after_end = min(
            now.astimezone(timezone.utc),
            closed + timedelta(days=self.settings.comparison_window_days),
        )
        if check_concurrent:
            concurrent = self.store.concurrent_closed_work_orders(
                work_order_id=work_order_id,
                asset_id=asset_id,
                start=before_start,
                end=after_end,
            )
            if concurrent:
                return {
                    "confirmed": None,
                    "reason": "concurrent_fix",
                    "notes": "Auto-verification blocked: another work order closed on this asset inside the comparison window.",
                    "sample_size_before": 0,
                    "sample_size_after": 0,
                    "gap_before_mean": None,
                    "gap_after_mean": None,
                    "p_value": None,
                    "retry": False,
                    "concurrent_work_order_ids": [row["work_order_id"] for row in concurrent],
                }

        before_rows = self.store.performance_readings_between(before_start, opened)
        after_rows = self.store.performance_readings_between(closed, after_end)
        matched_before, matched_after = match_operating_conditions(
            before_rows, after_rows, self.settings.operating_tolerance_pct,
        )
        before_mean = float(np.mean(matched_before)) if matched_before else None
        after_mean = float(np.mean(matched_after)) if matched_after else None
        sample_count = len(matched_before)
        if sample_count < self.settings.minimum_samples:
            can_retry = now < closed + timedelta(days=self.settings.comparison_window_days)
            return {
                "confirmed": None,
                "reason": "insufficient_matched_samples",
                "notes": (
                    f"Auto-verification inconclusive: {sample_count} matched readings; "
                    f"minimum is {self.settings.minimum_samples}."
                ),
                "sample_size_before": sample_count,
                "sample_size_after": len(matched_after),
                "gap_before_mean": before_mean,
                "gap_after_mean": after_mean,
                "p_value": None,
                "retry": can_retry,
            }

        test = ttest_rel(np.asarray(matched_before), np.asarray(matched_after), nan_policy="omit")
        p_value = float(test.pvalue) if math.isfinite(float(test.pvalue)) else 1.0
        improvement = float(before_mean - after_mean)
        significant = (
            p_value < self.settings.significance_alpha
            and abs(improvement) >= self.settings.minimum_effect_gap_pct
        )
        confirmed = improvement > 0 if significant else None
        if confirmed is True:
            reason = "significant_gap_improvement"
            notes = "Auto-confirmed: matched-condition generation gap improved significantly after closure."
        elif confirmed is False:
            reason = "significant_no_improvement"
            notes = "Auto-rejected: matched-condition generation gap significantly worsened after closure."
        else:
            reason = "statistically_inconclusive"
            notes = "Auto-verification inconclusive: before/after gap change was not statistically significant."
        return {
            "confirmed": confirmed,
            "reason": reason,
            "notes": notes,
            "sample_size_before": sample_count,
            "sample_size_after": len(matched_after),
            "gap_before_mean": round(before_mean, 6),
            "gap_after_mean": round(after_mean, 6),
            "p_value": round(p_value, 8),
            "retry": False,
        }

    def verify_job(self, job: dict, now: datetime) -> dict:
        result = self.evaluate(
            work_order_id=int(job["work_order_id"]),
            asset_id=str(job["asset_id"]),
            opened_at=str(job["opened_at"]),
            closed_at=str(job["closed_at"]),
            now=now,
        )
        feedback = self.store.upsert_auto_attribution_feedback(
            int(job["attribution_id"]),
            confirmed=result["confirmed"],
            notes=result["notes"],
            sample_size_before=result["sample_size_before"],
            sample_size_after=result["sample_size_after"],
            gap_before_mean=result["gap_before_mean"],
            gap_after_mean=result["gap_after_mean"],
            p_value=result["p_value"],
        )
        if feedback.get("manual_preserved"):
            result["reason"] = "manual_feedback_preserved"
            result["retry"] = False
        retry_at = (
            now + timedelta(days=self.settings.retry_days)
            if result["retry"] else None
        )
        self.store.finish_verification_job(
            int(job["job_id"]), reason=result["reason"], retry_at=retry_at,
        )
        return {**result, "feedback_id": feedback["feedback_id"]}

    def run_due(self, now: datetime | None = None) -> dict:
        current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        scheduled = self.store.sync_closed_work_order_verifications(self.settings.wait_days)
        jobs = self.store.due_verification_jobs(current)
        outcomes = [self.verify_job(job, current) for job in jobs]
        return {
            "scheduled": scheduled,
            "processed": len(outcomes),
            "confirmed": sum(item["confirmed"] is True for item in outcomes),
            "rejected": sum(item["confirmed"] is False for item in outcomes),
            "inconclusive": sum(item["confirmed"] is None for item in outcomes),
        }

    def monitoring(self) -> dict:
        return self.store.verification_monitoring(self.settings.null_alert_threshold)

    def backtest_manual_history(self) -> dict:
        rows = self.store.manual_feedback_work_orders()
        comparable = agreements = 0
        for row in rows:
            result = self.evaluate(
                work_order_id=int(row["work_order_id"]),
                asset_id=str(row["asset_id"]),
                opened_at=str(row["opened_at"]),
                closed_at=str(row["closed_at"]),
                now=_utc(row["closed_at"]) + timedelta(days=self.settings.comparison_window_days),
                check_concurrent=True,
            )
            if result["confirmed"] is None:
                continue
            comparable += 1
            agreements += result["confirmed"] == bool(row["confirmed"])
        agreement_rate = agreements / comparable if comparable else None
        return {
            "manual_rows": len(rows),
            "comparable_rows": comparable,
            "agreements": agreements,
            "agreement_rate": round(agreement_rate, 6) if agreement_rate is not None else None,
            "majority_reproduced": agreement_rate is not None and agreement_rate > 0.5,
            "status": "complete" if comparable else "no_comparable_manual_history",
        }


class AttributionVerificationScheduler:
    def __init__(self, service: AttributionVerificationService) -> None:
        self.service = service
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run(), name="attribution-auto-verification")

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
                summary = await asyncio.to_thread(self.service.run_due)
                if summary["processed"]:
                    logger.info("auto-verification processed summary=%s", summary)
                monitoring = self.service.monitoring()
                if monitoring["alert"]:
                    logger.warning(monitoring["alert"])
            except Exception:
                logger.exception("auto-verification scheduler failed")
            await asyncio.sleep(self.service.settings.scheduler_check_seconds)
