from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from .reference_curves import PerformanceCurveModel
from .store import Store

if TYPE_CHECKING:
    from .learned_attribution import LearnedAttributionService


logger = logging.getLogger("hydrovision.performance.attribution")
GRAVITY_M_S2 = 9.81


@dataclass(frozen=True)
class AttributionSettings:
    threshold_pct: float = 5.0
    evidence_window_seconds: int = 6 * 60 * 60
    meter_location: str = "generator_terminal"

    @property
    def enabled(self) -> bool:
        return self.meter_location == "generator_terminal"

    @classmethod
    def from_env(cls, performance_source: str) -> "AttributionSettings":
        configured_location = os.getenv("HYDROVISION_ACTUAL_MW_METER_LOCATION", "").strip().lower()
        if not configured_location:
            # The bundled mock explicitly represents generator output. Real
            # plant integration must name its metering boundary before Phase 4
            # is allowed to infer hydraulic causes.
            configured_location = "generator_terminal" if performance_source == "mock" else "unconfirmed"
        if configured_location not in {"generator_terminal", "grid_connection", "unconfirmed"}:
            raise ValueError(
                "HYDROVISION_ACTUAL_MW_METER_LOCATION must be "
                "'generator_terminal', 'grid_connection', or 'unconfirmed'"
            )
        threshold = float(os.getenv("HYDROVISION_ATTRIBUTION_GAP_THRESHOLD_PCT", "5"))
        window = int(os.getenv("HYDROVISION_ATTRIBUTION_EVIDENCE_WINDOW_SECONDS", "21600"))
        if threshold < 0 or window <= 0:
            raise ValueError("attribution threshold must be non-negative and evidence window positive")
        return cls(threshold_pct=threshold, evidence_window_seconds=window, meter_location=configured_location)


class AttributionService:
    """Evidence-linked rules with optional Phase 6 shadow/approved scoring."""

    def __init__(
        self,
        store: Store,
        curves: PerformanceCurveModel,
        settings: AttributionSettings,
        learned_service: "LearnedAttributionService | None" = None,
    ) -> None:
        self.store = store
        self.curves = curves
        self.settings = settings
        self.learned_service = learned_service
        if not settings.enabled:
            logger.warning(
                "gap attribution disabled actual_mw_meter_location=%s; "
                "confirm generator_terminal before enabling Phase 4",
                settings.meter_location,
            )

    def attribute_reading(self, reading_id: int) -> dict:
        reading = self.store.performance_reading(reading_id)
        if reading is None:
            raise KeyError(reading_id)
        gap_pct = reading.get("gap_pct")
        if gap_pct is None or float(gap_pct) <= self.settings.threshold_pct:
            return {"reading_id": reading_id, "status": "below_threshold", "created": 0}
        if not self.settings.enabled:
            return {"reading_id": reading_id, "status": "meter_location_unconfirmed", "created": 0}
        existing = self.store.attribution_run(reading_id)
        if existing:
            return {
                "reading_id": reading_id,
                "status": existing["status"],
                "created": 0,
                "deduplicated": True,
            }

        reading_ts = datetime.fromisoformat(str(reading["ts"])).astimezone(timezone.utc)
        events = self.store.closest_prior_active_detection_events(
            reading_ts,
            window_seconds=self.settings.evidence_window_seconds,
        )
        rules_by_type: dict[str, list[dict]] = {}
        for rule in self.store.attribution_rules():
            rules_by_type.setdefault(rule["defect_type"], []).append(rule)

        contributions_by_asset: dict[str, dict] = {}
        for event in events:
            for rule in rules_by_type.get(event["detection_type"], []):
                asset_ids = rule["params"].get("asset_ids", [])
                if asset_ids and event["asset_id"] not in asset_ids:
                    continue
                try:
                    estimated_loss = self._estimate_loss_mw(reading, event, rule)
                except (KeyError, TypeError, ValueError) as error:
                    logger.warning(
                        "attribution rule skipped reading_id=%s event_id=%s rule_id=%s error=%s",
                        reading_id, event["event_id"], rule["rule_id"], error,
                    )
                    continue
                if estimated_loss <= 0:
                    continue
                event_confidence = event.get("confidence")
                confidence = float(rule["confidence"])
                if event_confidence is not None:
                    confidence *= float(event_confidence)
                contribution = {
                    "asset_id": event["asset_id"],
                    "event_id": event["event_id"],
                    "estimated_loss_mw": round(min(estimated_loss, float(reading["theoretical_mw"])), 6),
                    "confidence": round(min(max(confidence, 0.0), 1.0), 4),
                }
                # Multiple active detector heads can describe the same site.
                # Keep the strongest evidence-linked estimate so rankings contain
                # one row per site and do not double-count correlated symptoms.
                previous = contributions_by_asset.get(event["asset_id"])
                if previous is None or (
                    contribution["estimated_loss_mw"], contribution["confidence"]
                ) > (previous["estimated_loss_mw"], previous["confidence"]):
                    contributions_by_asset[event["asset_id"]] = contribution

        contributions = list(contributions_by_asset.values())
        if contributions and self.learned_service is not None:
            # The learned service preserves the rule estimate, logs any shadow
            # estimate separately, and only replaces the displayed value when
            # a human-approved active model supports this specific defect type.
            contributions = self.learned_service.score_contributions(reading_id, contributions)

        self.store.save_attribution_run(
            reading_id,
            self.settings.threshold_pct,
            contributions,
        )
        if contributions:
            logger.info(
                "gap attribution complete reading_id=%s gap_pct=%.3f contributors=%s",
                reading_id, gap_pct, len(contributions),
            )
            status = "attributed"
        else:
            logger.warning(
                "gap attribution unexplained reading_id=%s gap_pct=%.3f "
                "evidence_window_seconds=%s",
                reading_id, gap_pct, self.settings.evidence_window_seconds,
            )
            status = "unexplained"
        return {"reading_id": reading_id, "status": status, "created": len(contributions)}

    def backfill(self) -> dict[str, int | str]:
        if not self.settings.enabled:
            return {"status": "disabled", "processed": 0, "attributed": 0, "unexplained": 0}
        rows = self.store.pending_attribution_readings(self.settings.threshold_pct)
        attributed = unexplained = 0
        for reading in rows:
            result = self.attribute_reading(int(reading["reading_id"]))
            attributed += result["status"] == "attributed"
            unexplained += result["status"] == "unexplained"
        return {
            "status": "complete",
            "processed": len(rows),
            "attributed": attributed,
            "unexplained": unexplained,
        }

    def on_reading_stored(self, reading_id: int) -> None:
        self.attribute_reading(reading_id)

    def _estimate_loss_mw(self, reading: dict, event: dict, rule: dict) -> float:
        params = rule["params"]
        operation = params["operation"]
        if rule["formula_type"] == "geometric" and operation == "rack_head_loss":
            return self._rack_head_loss(reading, event, params)
        if rule["formula_type"] == "geometric" and operation == "visual_gate_flow":
            return self._visual_gate_loss(reading, event)
        if rule["formula_type"] == "heuristic_map" and operation in {
            "flow_loss_pct", "efficiency_loss_pct",
        }:
            percentage = self._mapped_value(event, params)
            return float(reading["theoretical_mw"]) * percentage / 100
        if rule["formula_type"] == "heuristic_map" and operation == "head_reduction_m":
            head_reduction = self._mapped_value(event, params)
            return self._loss_from_head_reduction(reading, head_reduction)
        raise ValueError(f"unsupported formula_type/operation: {rule['formula_type']}/{operation}")

    def _base_hydraulics(self, reading: dict) -> tuple[float, float, float]:
        headwater = float(reading["headwater_level"])
        tailwater = float(reading["tailwater_level"])
        flow = self.curves.flow(float(reading["gate_position"]), headwater)
        loss = self.curves.loss(flow)
        net_head = headwater - tailwater - loss
        if net_head <= 0:
            raise ValueError("baseline net head is not positive")
        return flow, loss, net_head

    def _output_mw(self, flow: float, net_head: float) -> float:
        if flow <= 0 or net_head <= 0:
            return 0.0
        efficiency = self.curves.turbine_efficiency(flow, net_head)
        return 1000 * GRAVITY_M_S2 * flow * net_head * efficiency / 1_000_000

    def _rack_head_loss(self, reading: dict, event: dict, params: dict) -> float:
        blockage_pct = float(event["measurement"]["blockage_pct"])
        maximum = float(params["maximum_blockage_fraction"])
        blockage = min(max(blockage_pct / 100, 0.0), maximum)
        if blockage <= 0:
            return 0.0
        flow, _, net_head = self._base_hydraulics(reading)
        open_area = float(params["rack_open_area_m2"])
        coefficient = float(params["loss_coefficient"])
        if open_area <= 0 or coefficient < 0:
            raise ValueError("rack area/coefficient configuration is invalid")
        clean_velocity = flow / open_area
        blocked_velocity = flow / (open_area * (1 - blockage))
        # Darcy-style velocity-head relation: the configurable rack coefficient
        # multiplies the incremental velocity head caused by lost open area.
        extra_loss_m = coefficient * (
            blocked_velocity ** 2 - clean_velocity ** 2
        ) / (2 * GRAVITY_M_S2)
        # Keep the Phase 2 baseline efficiency for this differential head-loss
        # estimate. Severe blockage can move adjusted head outside the OEM hill
        # curve domain; extrapolating efficiency there would be less defensible.
        baseline_efficiency = self.curves.turbine_efficiency(flow, net_head)
        adjusted_head = max(0.0, net_head - extra_loss_m)
        damaged_output = (
            1000 * GRAVITY_M_S2 * flow * adjusted_head * baseline_efficiency / 1_000_000
        )
        return max(0.0, float(reading["theoretical_mw"]) - damaged_output)

    def _visual_gate_loss(self, reading: dict, event: dict) -> float:
        visual_position = event["measurement"].get("visual_gate_position_pct")
        if visual_position is None:
            raise ValueError("gate event has no visual position")
        headwater = float(reading["headwater_level"])
        tailwater = float(reading["tailwater_level"])
        visual_flow = self.curves.flow(float(visual_position), headwater)
        visual_loss = self.curves.loss(visual_flow)
        visual_output = self._output_mw(visual_flow, headwater - tailwater - visual_loss)
        return max(0.0, float(reading["theoretical_mw"]) - visual_output)

    @staticmethod
    def _mapped_value(event: dict, params: dict) -> float:
        mapping = params["severity_map"]
        value = float(mapping.get(event["severity"], 0.0))
        if params.get("scale_by_event_confidence") and event.get("confidence") is not None:
            value *= float(event["confidence"])
        return max(0.0, value)

    def _loss_from_head_reduction(self, reading: dict, head_reduction_m: float) -> float:
        flow, _, net_head = self._base_hydraulics(reading)
        baseline_efficiency = self.curves.turbine_efficiency(flow, net_head)
        adjusted_head = max(0.0, net_head - head_reduction_m)
        damaged_output = (
            1000 * GRAVITY_M_S2 * flow * adjusted_head * baseline_efficiency / 1_000_000
        )
        return max(0.0, float(reading["theoretical_mw"]) - damaged_output)
