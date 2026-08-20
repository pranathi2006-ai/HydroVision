from __future__ import annotations

import csv
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import RegularGridInterpolator, interp1d

from .store import Store

if TYPE_CHECKING:
    from .performance import PerformanceReading


logger = logging.getLogger("hydrovision.performance.calculation")


class ReferenceCurveError(ValueError):
    pass


class PerformanceCalculationError(ValueError):
    pass


@dataclass(frozen=True)
class ReferenceCurveData:
    turbine: list[dict]
    gate: list[dict]
    loss: list[dict]


@dataclass(frozen=True)
class CalculatedPerformance:
    theoretical_mw: float
    gap_pct: float
    flow_m3s: float
    loss_m: float
    net_head_m: float
    efficiency: float


def _float(row: dict, column: str, source: str) -> float:
    try:
        value = float(row[column])
    except (KeyError, TypeError, ValueError) as error:
        raise ReferenceCurveError(f"{source}: invalid or missing {column}") from error
    if not math.isfinite(value):
        raise ReferenceCurveError(f"{source}: {column} must be finite")
    return value


def _normalized_data(payload: dict, source: str) -> ReferenceCurveData:
    try:
        turbine_rows = payload["turbine_performance_curve"]
        gate_rows = payload["gate_flow_curve"]
        loss_rows = payload["hydraulic_loss_baseline"]
    except KeyError as error:
        raise ReferenceCurveError(f"{source}: missing curve collection {error.args[0]}") from error
    if not all(isinstance(rows, list) for rows in (turbine_rows, gate_rows, loss_rows)):
        raise ReferenceCurveError(f"{source}: every curve collection must be an array")

    turbine = [
        {
            "unit_id": str(row.get("unit_id", "")).strip(),
            "flow_m3s": _float(row, "flow_m3s", source),
            "head_m": _float(row, "head_m", source),
            "efficiency": _float(row, "efficiency", source),
        }
        for row in turbine_rows
    ]
    gate = [
        {
            "gate_position": _float(row, "gate_position", source),
            "head_m": _float(row, "head_m", source),
            "flow_m3s": _float(row, "flow_m3s", source),
        }
        for row in gate_rows
    ]
    loss = [
        {
            "flow_m3s": _float(row, "flow_m3s", source),
            "loss_m": _float(row, "loss_m", source),
        }
        for row in loss_rows
    ]
    _validate_rows(turbine, gate, loss, source)
    return ReferenceCurveData(turbine=turbine, gate=gate, loss=loss)


def _validate_rectangular_grid(rows: list[dict], x: str, y: str, source: str) -> None:
    x_values = {row[x] for row in rows}
    y_values = {row[y] for row in rows}
    points = {(row[x], row[y]) for row in rows}
    if len(x_values) < 2 or len(y_values) < 2:
        raise ReferenceCurveError(f"{source}: {x}/{y} curve needs at least a 2x2 grid")
    if len(points) != len(rows):
        raise ReferenceCurveError(f"{source}: duplicate {x}/{y} curve point")
    if len(points) != len(x_values) * len(y_values):
        raise ReferenceCurveError(f"{source}: {x}/{y} curve must be a complete rectangular grid")


def _validate_rows(turbine: list[dict], gate: list[dict], loss: list[dict], source: str) -> None:
    if not turbine or not gate or len(loss) < 2:
        raise ReferenceCurveError(f"{source}: all three curve collections are required")
    if any(not row["unit_id"] for row in turbine):
        raise ReferenceCurveError(f"{source}: turbine unit_id cannot be blank")
    if any(not 0 <= row["efficiency"] <= 1 for row in turbine):
        raise ReferenceCurveError(f"{source}: efficiency must be between 0.0 and 1.0")
    if any(row["flow_m3s"] < 0 for row in turbine + gate):
        raise ReferenceCurveError(f"{source}: flow_m3s cannot be negative")
    if any(row["loss_m"] < 0 for row in loss):
        raise ReferenceCurveError(f"{source}: loss_m cannot be negative")
    for unit_id in {row["unit_id"] for row in turbine}:
        unit_rows = [row for row in turbine if row["unit_id"] == unit_id]
        _validate_rectangular_grid(unit_rows, "flow_m3s", "head_m", source)
    _validate_rectangular_grid(gate, "gate_position", "head_m", source)
    loss_flows = [row["flow_m3s"] for row in loss]
    if len(set(loss_flows)) != len(loss_flows):
        raise ReferenceCurveError(f"{source}: duplicate hydraulic-loss flow point")


def load_reference_curve_file(path: Path) -> ReferenceCurveData:
    path = path.resolve()
    if path.is_file() and path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ReferenceCurveError(f"{path}: JSON root must be an object")
        return _normalized_data(payload, str(path))
    if not path.is_dir():
        raise ReferenceCurveError(f"Reference curve path does not exist: {path}")

    filenames = {
        "turbine_performance_curve": "turbine_performance_curve.csv",
        "gate_flow_curve": "gate_flow_curve.csv",
        "hydraulic_loss_baseline": "hydraulic_loss_baseline.csv",
    }
    payload: dict[str, list[dict]] = {}
    for key, filename in filenames.items():
        csv_path = path / filename
        if not csv_path.is_file():
            raise ReferenceCurveError(f"Missing reference curve file: {csv_path}")
        with csv_path.open(newline="", encoding="utf-8-sig") as handle:
            payload[key] = list(csv.DictReader(handle))
    return _normalized_data(payload, str(path))


def import_reference_curves(
    store: Store,
    path: Path,
    *,
    dataset_name: str,
    is_demo: bool = False,
) -> ReferenceCurveData:
    data = load_reference_curve_file(path)
    store.replace_reference_curves(
        data.turbine,
        data.gate,
        data.loss,
        dataset_name=dataset_name,
        source_path=str(path.resolve()),
        is_demo=is_demo,
    )
    logger.info(
        "reference curves imported dataset=%s turbine_points=%s gate_points=%s loss_points=%s",
        dataset_name,
        len(data.turbine),
        len(data.gate),
        len(data.loss),
    )
    return data


def _regular_grid(rows: list[dict], x_key: str, y_key: str, value_key: str):
    x_values = np.asarray(sorted({row[x_key] for row in rows}), dtype=float)
    y_values = np.asarray(sorted({row[y_key] for row in rows}), dtype=float)
    values = np.empty((len(x_values), len(y_values)), dtype=float)
    x_index = {value: index for index, value in enumerate(x_values)}
    y_index = {value: index for index, value in enumerate(y_values)}
    for row in rows:
        values[x_index[row[x_key]], y_index[row[y_key]]] = row[value_key]
    return RegularGridInterpolator(
        (x_values, y_values),
        values,
        method="linear",
        bounds_error=True,
    )


class PerformanceCurveModel:
    """In-memory SciPy interpolators built once from the database curves."""

    def __init__(self, store: Store, unit_id: str) -> None:
        curves = store.reference_curve_rows()
        turbine_rows = [row for row in curves["turbine"] if row["unit_id"] == unit_id]
        if not turbine_rows:
            raise ReferenceCurveError(f"No turbine performance curve found for unit_id={unit_id}")
        _validate_rows(turbine_rows, curves["gate"], curves["loss"], "database reference curves")
        self.unit_id = unit_id
        self.gate_flow = _regular_grid(curves["gate"], "gate_position", "head_m", "flow_m3s")
        self.efficiency = _regular_grid(turbine_rows, "flow_m3s", "head_m", "efficiency")
        loss_rows = sorted(curves["loss"], key=lambda row: row["flow_m3s"])
        self.hydraulic_loss = interp1d(
            [row["flow_m3s"] for row in loss_rows],
            [row["loss_m"] for row in loss_rows],
            kind="linear",
            bounds_error=True,
        )

    @staticmethod
    def _scalar(value) -> float:
        return float(np.asarray(value).reshape(-1)[0])

    def flow(self, gate_position: float, headwater_level: float) -> float:
        return self._scalar(self.gate_flow((gate_position, headwater_level)))

    def loss(self, flow_m3s: float) -> float:
        return self._scalar(self.hydraulic_loss(flow_m3s))

    def turbine_efficiency(self, flow_m3s: float, net_head_m: float) -> float:
        return self._scalar(self.efficiency((flow_m3s, net_head_m)))


class PerformanceCalculationService:
    def __init__(
        self,
        store: Store,
        curves: PerformanceCurveModel,
        *,
        nameplate_capacity_mw: float,
    ) -> None:
        if nameplate_capacity_mw <= 0:
            raise ValueError("nameplate_capacity_mw must be positive")
        self.store = store
        self.curves = curves
        self.nameplate_capacity_mw = nameplate_capacity_mw

    def calculate(self, reading: "PerformanceReading" | dict) -> CalculatedPerformance:
        def value(name: str) -> float:
            raw = reading[name] if isinstance(reading, dict) else getattr(reading, name)
            if raw is None or not math.isfinite(float(raw)):
                raise PerformanceCalculationError(f"{name} is missing or non-finite")
            return float(raw)

        gate_position = value("gate_position")
        headwater_level = value("headwater_level")
        tailwater_level = value("tailwater_level")
        actual_mw = value("actual_mw")
        try:
            flow_m3s = self.curves.flow(gate_position, headwater_level)
            loss_m = self.curves.loss(flow_m3s)
            net_head_m = headwater_level - tailwater_level - loss_m
            if net_head_m <= 0:
                raise PerformanceCalculationError(f"computed net head is not positive: {net_head_m}")
            efficiency = self.curves.turbine_efficiency(flow_m3s, net_head_m)
        except ValueError as error:
            raise PerformanceCalculationError(f"reference curve lookup failed: {error}") from error

        theoretical_mw = (1000 * 9.81 * flow_m3s * net_head_m * efficiency) / 1_000_000
        if not math.isfinite(theoretical_mw):
            raise PerformanceCalculationError("computed theoretical_mw is non-finite")
        if theoretical_mw < 0 or theoretical_mw > self.nameplate_capacity_mw:
            clamped = min(max(theoretical_mw, 0.0), self.nameplate_capacity_mw)
            logger.warning(
                "implausible theoretical output clamped raw_theoretical_mw=%.6f clamped_mw=%.6f nameplate_mw=%.6f",
                theoretical_mw,
                clamped,
                self.nameplate_capacity_mw,
            )
            theoretical_mw = clamped
        if theoretical_mw <= 0:
            raise PerformanceCalculationError("theoretical_mw is zero; gap_pct is undefined")
        gap_pct = (theoretical_mw - actual_mw) / theoretical_mw * 100
        if not math.isfinite(gap_pct):
            raise PerformanceCalculationError("computed gap_pct is non-finite")
        return CalculatedPerformance(
            theoretical_mw=round(theoretical_mw, 6),
            gap_pct=round(gap_pct, 6),
            flow_m3s=round(flow_m3s, 6),
            loss_m=round(loss_m, 6),
            net_head_m=round(net_head_m, 6),
            efficiency=round(efficiency, 8),
        )

    def backfill(self, *, overwrite: bool = False) -> dict[str, int]:
        rows = self.store.performance_readings_for_calculation(only_missing=not overwrite)
        updated = 0
        failed = 0
        for row in rows:
            try:
                calculated = self.calculate(row)
            except PerformanceCalculationError as error:
                failed += 1
                logger.error(
                    "performance backfill failed reading_id=%s error=%s",
                    row["reading_id"],
                    error,
                )
                continue
            self.store.update_performance_calculation(row["reading_id"], calculated)
            updated += 1
        summary = {"processed": len(rows), "updated": updated, "failed": failed}
        logger.info("performance backfill complete %s", summary)
        return summary
