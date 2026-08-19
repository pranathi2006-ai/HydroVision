from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Protocol

from .store import Store


logger = logging.getLogger("hydrovision.performance")


def parse_timestamp(value: object) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def optional_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class PerformanceReading:
    ts: datetime | None
    headwater_level: float | None
    tailwater_level: float | None
    gate_position: float | None
    actual_mw: float | None


class SourceAdapter(Protocol):
    """The single contract used by performance ingestion adapters."""

    def getLatestReading(self) -> PerformanceReading:
        ...


class MockSourceAdapter:
    """Produces stable, smooth signals with realistic hydropower relationships."""

    def __init__(self, clock=lambda: datetime.now(timezone.utc)) -> None:
        self._clock = clock

    def getLatestReading(self) -> PerformanceReading:
        now = self._clock().astimezone(timezone.utc)
        elapsed_minutes = now.timestamp() / 60

        # These are synthetic operating signals, not a theoretical-generation
        # calculation. Slow harmonics avoid random jumps that would mislead
        # consumers developed against the mock source.
        dispatch = math.sin(2 * math.pi * elapsed_minutes / (12 * 60))
        governor = math.sin(2 * math.pi * elapsed_minutes / 95)
        reservoir = math.sin(2 * math.pi * elapsed_minutes / (36 * 60))

        gate_position = 61.0 + 10.5 * dispatch + 1.4 * governor
        headwater_level = 118.35 + 0.32 * reservoir - 0.04 * dispatch
        tailwater_level = 82.10 + 0.18 * dispatch + 0.04 * governor
        actual_mw = 48.0 + 8.6 * dispatch + 1.1 * governor + 0.25 * reservoir

        return PerformanceReading(
            ts=now,
            headwater_level=round(headwater_level, 3),
            tailwater_level=round(tailwater_level, 3),
            gate_position=round(gate_position, 2),
            actual_mw=round(actual_mw, 3),
        )


class RealSourceAdapter:
    """HTTP plant-source adapter; endpoint and field mapping are configuration.

    Replace only this transport body if the confirmed integration is OPC-UA or
    a historian SDK. The ingestion service and SourceAdapter contract stay the
    same.
    """

    def __init__(
        self,
        endpoint: str,
        *,
        token: str = "",
        timeout_seconds: float = 15,
        field_map: dict[str, str] | None = None,
    ) -> None:
        self.endpoint = endpoint
        self.token = token
        self.timeout_seconds = timeout_seconds
        self.field_map = field_map or {
            "ts": "timestamp",
            "headwater_level": "headwater_level",
            "tailwater_level": "tailwater_level",
            "gate_position": "gate_position",
            "actual_mw": "generation_mw",
        }

    def getLatestReading(self) -> PerformanceReading:
        if not self.endpoint:
            raise RuntimeError(
                "RealSourceAdapter is active but HYDROVISION_PLANT_API_URL is not configured"
            )
        headers = {"Accept": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        request = urllib.request.Request(self.endpoint, headers=headers)
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            payload = json.load(response)
        if not isinstance(payload, dict):
            raise ValueError("Plant endpoint must return one JSON object")

        return PerformanceReading(
            ts=parse_timestamp(payload.get(self.field_map["ts"])),
            headwater_level=optional_float(payload.get(self.field_map["headwater_level"])),
            tailwater_level=optional_float(payload.get(self.field_map["tailwater_level"])),
            gate_position=optional_float(payload.get(self.field_map["gate_position"])),
            actual_mw=optional_float(payload.get(self.field_map["actual_mw"])),
        )


@dataclass(frozen=True)
class PerformanceSettings:
    source: str = "mock"
    poll_interval_seconds: int = 300
    source_update_seconds: int = 300
    headwater_min: float = 0
    headwater_max: float = 500
    tailwater_min: float = 0
    tailwater_max: float = 500
    gate_position_min: float = 0
    gate_position_max: float = 100
    actual_mw_min: float = 0
    actual_mw_max: float = 1000

    @property
    def effective_interval_seconds(self) -> int:
        return max(self.poll_interval_seconds, self.source_update_seconds)

    @classmethod
    def from_env(cls) -> "PerformanceSettings":
        return cls(
            source=os.getenv("HYDROVISION_PERFORMANCE_SOURCE", "mock").strip().lower(),
            poll_interval_seconds=max(1, int(os.getenv("HYDROVISION_PERFORMANCE_POLL_SECONDS", "300"))),
            source_update_seconds=max(1, int(os.getenv("HYDROVISION_SOURCE_UPDATE_SECONDS", "300"))),
            headwater_min=float(os.getenv("HYDROVISION_HEADWATER_MIN", "0")),
            headwater_max=float(os.getenv("HYDROVISION_HEADWATER_MAX", "500")),
            tailwater_min=float(os.getenv("HYDROVISION_TAILWATER_MIN", "0")),
            tailwater_max=float(os.getenv("HYDROVISION_TAILWATER_MAX", "500")),
            gate_position_min=float(os.getenv("HYDROVISION_GATE_POSITION_MIN", "0")),
            gate_position_max=float(os.getenv("HYDROVISION_GATE_POSITION_MAX", "100")),
            actual_mw_min=float(os.getenv("HYDROVISION_ACTUAL_MW_MIN", "0")),
            actual_mw_max=float(os.getenv("HYDROVISION_ACTUAL_MW_MAX", "1000")),
        )


def build_adapter(settings: PerformanceSettings) -> SourceAdapter:
    if settings.source == "mock":
        return MockSourceAdapter()
    if settings.source == "real":
        field_map = {
            "ts": os.getenv("HYDROVISION_PLANT_FIELD_TS", "timestamp"),
            "headwater_level": os.getenv("HYDROVISION_PLANT_FIELD_HEADWATER", "headwater_level"),
            "tailwater_level": os.getenv("HYDROVISION_PLANT_FIELD_TAILWATER", "tailwater_level"),
            "gate_position": os.getenv("HYDROVISION_PLANT_FIELD_GATE", "gate_position"),
            "actual_mw": os.getenv("HYDROVISION_PLANT_FIELD_ACTUAL_MW", "generation_mw"),
        }
        return RealSourceAdapter(
            os.getenv("HYDROVISION_PLANT_API_URL", ""),
            token=os.getenv("HYDROVISION_PLANT_API_TOKEN", ""),
            timeout_seconds=float(os.getenv("HYDROVISION_PLANT_API_TIMEOUT_SECONDS", "15")),
            field_map=field_map,
        )
    raise ValueError("HYDROVISION_PERFORMANCE_SOURCE must be 'mock' or 'real'")


class PerformanceIngestionService:
    def __init__(
        self,
        store: Store,
        adapter: SourceAdapter,
        settings: PerformanceSettings,
    ) -> None:
        self.store = store
        self.adapter = adapter
        self.settings = settings
        self._task: asyncio.Task[None] | None = None
        self._warned_gap_after: datetime | None = None

    def validate(self, reading: PerformanceReading, *, now: datetime | None = None) -> list[str]:
        now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        errors: list[str] = []
        if reading.ts is None:
            errors.append("ts is missing or is not timezone-aware ISO-8601")
        elif reading.ts.tzinfo is None:
            errors.append("ts must include a timezone offset")
        elif reading.ts > now + timedelta(seconds=self.settings.effective_interval_seconds):
            errors.append("ts is unexpectedly in the future")

        ranges = (
            ("headwater_level", reading.headwater_level, self.settings.headwater_min, self.settings.headwater_max),
            ("tailwater_level", reading.tailwater_level, self.settings.tailwater_min, self.settings.tailwater_max),
            ("gate_position", reading.gate_position, self.settings.gate_position_min, self.settings.gate_position_max),
            ("actual_mw", reading.actual_mw, self.settings.actual_mw_min, self.settings.actual_mw_max),
        )
        for name, value, minimum, maximum in ranges:
            if value is None or not math.isfinite(value):
                errors.append(f"{name} is missing or non-finite")
            elif not minimum <= value <= maximum:
                errors.append(f"{name}={value} is outside [{minimum}, {maximum}]")
        return errors

    def _log_gap_if_needed(self, now: datetime, last_successful: datetime | None) -> None:
        if last_successful is None:
            return
        threshold = timedelta(seconds=2 * self.settings.effective_interval_seconds)
        if now - last_successful <= threshold or self._warned_gap_after == last_successful:
            return
        expected_next = last_successful + timedelta(seconds=self.settings.effective_interval_seconds)
        logger.warning(
            "performance reading gap detected missing_window_start=%s missing_window_end=%s last_successful=%s",
            expected_next.isoformat(),
            now.isoformat(),
            last_successful.isoformat(),
        )
        self._warned_gap_after = last_successful

    def poll_once(self, *, now: datetime | None = None) -> bool:
        checked_at = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        last_successful = self.store.latest_performance_timestamp()
        self._log_gap_if_needed(checked_at, last_successful)
        try:
            reading = self.adapter.getLatestReading()
        except Exception:
            logger.exception("performance source poll failed adapter=%s", type(self.adapter).__name__)
            return False

        errors = self.validate(reading, now=checked_at)
        if (
            reading.ts is not None
            and reading.ts.tzinfo is not None
            and last_successful is not None
            and reading.ts <= last_successful
        ):
            errors.append(f"ts={reading.ts.isoformat()} is stale or duplicated")
        if errors:
            logger.error(
                "performance reading rejected adapter=%s errors=%s reading=%s",
                type(self.adapter).__name__,
                "; ".join(errors),
                asdict(reading),
            )
            return False

        assert reading.ts is not None
        self._log_gap_if_needed(reading.ts, last_successful)
        reading_id = self.store.insert_performance_reading(reading)
        self._warned_gap_after = None
        logger.info(
            "performance reading stored reading_id=%s ts=%s actual_mw=%.3f headwater_level=%.3f tailwater_level=%.3f gate_position=%.2f adapter=%s",
            reading_id,
            reading.ts.isoformat(),
            reading.actual_mw,
            reading.headwater_level,
            reading.tailwater_level,
            reading.gate_position,
            type(self.adapter).__name__,
        )
        return True

    async def _run(self) -> None:
        interval = self.settings.effective_interval_seconds
        logger.info(
            "performance ingestion started adapter=%s interval_seconds=%s",
            type(self.adapter).__name__,
            interval,
        )
        while True:
            await asyncio.to_thread(self.poll_once)
            await asyncio.sleep(interval)

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run(), name="performance-ingestion")

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None
