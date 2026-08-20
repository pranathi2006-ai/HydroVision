from __future__ import annotations

import hashlib
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from .detector import LocalDetector

if TYPE_CHECKING:
    from .store import Store


LOCATION_TO_ASSET = {
    "turbine-a": "turbine_1",
    "turbine-b": "turbine_2",
    "penstock": "penstock_valve",
    "transformer": "main_transformer",
    "intake": "intake_gate",
    "draft-tube": "draft_tube",
}

DEFAULT_SENSOR = {
    "turbine_1": "turbine_1_camera",
    "turbine_2": "turbine_2_camera",
    "penstock_valve": "penstock_valve_camera",
    "main_transformer": "main_transformer_thermal",
    "intake_gate": "intake_gate_camera",
    "draft_tube": "draft_tube_camera",
}


@dataclass(frozen=True)
class SiteDetection:
    detection_type: str
    defect_present: bool
    severity: str
    confidence: float | None
    bbox: tuple[int, int, int, int] | None
    measurement: dict
    engine: str

    def row(self) -> dict:
        data = asdict(self)
        data["bbox"] = list(self.bbox) if self.bbox is not None else None
        return data


def severity_for_percentage(value: float, warning: float, critical: float) -> str:
    if value >= critical:
        return "critical"
    if value >= warning:
        return "warning"
    return "normal"


def _largest_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    x, y, width, height = cv2.boundingRect(max(contours, key=cv2.contourArea))
    return x, y, x + width, y + height


class AreaBlockageDetector:
    """Local blockage segmentation with an optional task-specific YOLO model."""

    def __init__(self, detection_type: str, threshold_pct: float, model_env: str) -> None:
        self.detection_type = detection_type
        self.threshold_pct = threshold_pct
        self.model = None
        model_path = os.getenv(model_env)
        if model_path and Path(model_path).is_file():
            from ultralytics import YOLO

            self.model = YOLO(model_path)

    def predict(self, image: np.ndarray) -> SiteDetection:
        if self.model is not None:
            result = self.model.predict(image, imgsz=1024, conf=0.25, device="cpu", verbose=False)[0]
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            if result.masks is not None:
                for polygon in result.masks.xy:
                    cv2.fillPoly(mask, [np.asarray(polygon, dtype=np.int32)], 255)
            engine = "local-yolo-segmentation"
        else:
            # Adjacent-domain baseline: compact, brown/green, low-value debris
            # regions. Thin rack bars are excluded by contour geometry.
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            brown_green = cv2.inRange(hsv, np.array([4, 45, 15]), np.array([95, 255, 205]))
            dark = cv2.inRange(hsv[:, :, 2], 0, 55)
            candidate = cv2.bitwise_or(brown_green, dark)
            candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
            candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
            mask = np.zeros_like(candidate)
            image_area = image.shape[0] * image.shape[1]
            contours, _ = cv2.findContours(candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                x, y, width, height = cv2.boundingRect(contour)
                if area < image_area * 0.001 or min(width, height) < 8:
                    continue
                if max(width / max(height, 1), height / max(width, 1)) > 10:
                    continue
                cv2.drawContours(mask, [contour], -1, 255, thickness=-1)
            engine = "local-cv-blockage-baseline"

        blockage_pct = round(float(np.count_nonzero(mask)) / max(1, mask.size) * 100, 2)
        present = blockage_pct >= self.threshold_pct
        confidence = min(0.97, 0.55 + blockage_pct / 100) if present else min(0.95, 0.55 + (self.threshold_pct - blockage_pct) / max(self.threshold_pct, 1) * 0.35)
        return SiteDetection(
            detection_type=self.detection_type,
            defect_present=present,
            severity=severity_for_percentage(blockage_pct, self.threshold_pct, self.threshold_pct * 2.5),
            confidence=round(confidence, 4),
            bbox=_largest_bbox(mask) if present else None,
            measurement={
                "blockage_pct": blockage_pct,
                "threshold_pct": self.threshold_pct,
                "status": "blocked" if present else "healthy",
            },
            engine=engine,
        )


class GatePositionVerifier:
    """Fixed-camera geometric gate position measurement; no learned model."""

    def __init__(self) -> None:
        self.mismatch_threshold = float(os.getenv("HYDROVISION_GATE_MISMATCH_THRESHOLD_PCT", "8"))
        self.open_edge_ratio = float(os.getenv("HYDROVISION_GATE_OPEN_EDGE_RATIO", "0.20"))
        self.closed_edge_ratio = float(os.getenv("HYDROVISION_GATE_CLOSED_EDGE_RATIO", "0.80"))

    def predict(self, image: np.ndarray, commanded_pct: float | None) -> SiteDetection:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 140)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=max(35, image.shape[1] // 8),
            minLineLength=max(35, image.shape[1] // 3), maxLineGap=20,
        )
        candidates: list[tuple[int, int, int, int]] = []
        if lines is not None:
            for raw in lines[:, 0]:
                x1, y1, x2, y2 = (int(value) for value in raw)
                if abs(y2 - y1) <= max(3, int(abs(x2 - x1) * 0.06)):
                    candidates.append((x1, y1, x2, y2))
        if candidates:
            line = max(candidates, key=lambda item: abs(item[2] - item[0]))
            edge_y = (line[1] + line[3]) / 2
            open_y = image.shape[0] * self.open_edge_ratio
            closed_y = image.shape[0] * self.closed_edge_ratio
            visual_pct = float(np.clip((closed_y - edge_y) / max(1.0, closed_y - open_y) * 100, 0, 100))
            geometry_valid = True
            bbox = (min(line[0], line[2]), max(0, int(edge_y) - 3), max(line[0], line[2]), min(image.shape[0], int(edge_y) + 3))
        else:
            edge_y = None
            visual_pct = None
            geometry_valid = False
            bbox = None

        mismatch = abs(visual_pct - commanded_pct) if visual_pct is not None and commanded_pct is not None else None
        present = mismatch is not None and mismatch > self.mismatch_threshold
        measurement = {
            "visual_gate_position_pct": round(visual_pct, 2) if visual_pct is not None else None,
            "commanded_gate_position_pct": round(commanded_pct, 2) if commanded_pct is not None else None,
            "mismatch_pct_points": round(mismatch, 2) if mismatch is not None else None,
            "mismatch_threshold_pct_points": self.mismatch_threshold,
            "detected_edge_y_px": round(edge_y, 1) if edge_y is not None else None,
            "geometry_valid": geometry_valid,
            "status": "mismatch" if present else ("healthy" if mismatch is not None else "unavailable"),
        }
        confidence = None if not geometry_valid else round(min(0.98, 0.70 + (abs(line[2] - line[0]) / image.shape[1]) * 0.25), 4)
        return SiteDetection(
            detection_type="gate_position_mismatch",
            defect_present=present,
            severity="warning" if present else "normal",
            confidence=confidence,
            bbox=bbox,
            measurement=measurement,
            engine="local-cv-gate-geometry",
        )


class CavitationWearDetector:
    """Local pitting-area detector with optional turbine-specific weights."""

    def __init__(self) -> None:
        self.threshold_pct = float(os.getenv("HYDROVISION_CAVITATION_WEAR_THRESHOLD_PCT", "0.8"))
        self.model = None
        path = os.getenv("HYDROVISION_CAVITATION_MODEL_PATH")
        if path and Path(path).is_file():
            from ultralytics import YOLO

            self.model = YOLO(path)

    def predict(self, image: np.ndarray) -> SiteDetection:
        if self.model is not None:
            result = self.model.predict(image, imgsz=1024, conf=0.25, device="cpu", verbose=False)[0]
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            if result.masks is not None:
                for polygon in result.masks.xy:
                    cv2.fillPoly(mask, [np.asarray(polygon, dtype=np.int32)], 255)
            engine = "local-yolo-cavitation"
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            normalized = cv2.createCLAHE(2.0, (8, 8)).apply(gray)
            blackhat = cv2.morphologyEx(normalized, cv2.MORPH_BLACKHAT, np.ones((13, 13), np.uint8))
            _, candidate = cv2.threshold(blackhat, 24, 255, cv2.THRESH_BINARY)
            mask = np.zeros_like(candidate)
            image_area = mask.size
            contours, _ = cv2.findContours(candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if 6 <= area <= image_area * 0.02:
                    x, y, width, height = cv2.boundingRect(contour)
                    aspect = width / max(height, 1)
                    if 0.25 <= aspect <= 4:
                        cv2.drawContours(mask, [contour], -1, 255, -1)
            engine = "local-cv-pitting-baseline"
        pitting_pct = round(np.count_nonzero(mask) / max(1, mask.size) * 100, 2)
        present = pitting_pct >= self.threshold_pct
        return SiteDetection(
            detection_type="cavitation_wear",
            defect_present=present,
            severity=severity_for_percentage(pitting_pct, self.threshold_pct, self.threshold_pct * 4),
            confidence=round(min(0.95, 0.55 + abs(pitting_pct - self.threshold_pct) / max(5, self.threshold_pct) * 0.3), 4),
            bbox=_largest_bbox(mask) if present else None,
            measurement={
                "pitting_area_pct": pitting_pct,
                "threshold_pct": self.threshold_pct,
                "status": "wear_detected" if present else "healthy",
            },
            engine=engine,
        )


class ThermalHotspotDetector:
    """Rule-based apparent-temperature delta with consecutive-frame persistence."""

    def __init__(self) -> None:
        self.threshold_c = float(os.getenv("HYDROVISION_THERMAL_DELTA_THRESHOLD_C", "12"))
        self.minimum_frames = int(os.getenv("HYDROVISION_THERMAL_PERSISTENCE_FRAMES", "3"))
        self.minimum_c = float(os.getenv("HYDROVISION_THERMAL_SCALE_MIN_C", "20"))
        self.maximum_c = float(os.getenv("HYDROVISION_THERMAL_SCALE_MAX_C", "100"))
        self._counts: dict[str, int] = {}

    def predict(self, image: np.ndarray, asset_id: str, persisted_count: int = 0) -> SiteDetection:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        baseline_value = float(np.percentile(gray, 50))
        peak_value = float(np.percentile(gray, 99.5))
        scale = (self.maximum_c - self.minimum_c) / 255
        baseline_c = self.minimum_c + baseline_value * scale
        peak_c = self.minimum_c + peak_value * scale
        delta_c = max(0.0, peak_c - baseline_c)
        above = delta_c >= self.threshold_c
        previous = max(self._counts.get(asset_id, 0), persisted_count)
        count = previous + 1 if above else 0
        self._counts[asset_id] = count
        present = above and count >= self.minimum_frames
        threshold_value = baseline_value + self.threshold_c / max(scale, 1e-6)
        mask = np.where(gray >= threshold_value, 255, 0).astype(np.uint8)
        return SiteDetection(
            detection_type="thermal_hotspot",
            defect_present=present,
            severity="critical" if present and delta_c >= self.threshold_c * 2 else ("warning" if present else "normal"),
            confidence=round(min(0.99, 0.60 + delta_c / max(self.threshold_c, 1) * 0.15), 4),
            bbox=_largest_bbox(mask) if present else None,
            measurement={
                "baseline_temp_c": round(baseline_c, 2),
                "peak_temp_c": round(peak_c, 2),
                "delta_t_c": round(delta_c, 2),
                "threshold_delta_t_c": self.threshold_c,
                "consecutive_hot_frames": count,
                "required_consecutive_frames": self.minimum_frames,
                "radiometric_calibration": "configured_linear_scale",
                "status": "hotspot" if present else ("pending_persistence" if above else "healthy"),
            },
            engine="local-cv-thermal-rule",
        )


class SiteDetectionService:
    def __init__(self, store: "Store", existing_detector: LocalDetector | None = None) -> None:
        self.store = store
        self.existing = existing_detector or LocalDetector()
        self.intake_blockage = AreaBlockageDetector(
            "trash_rack_blockage", float(os.getenv("HYDROVISION_TRASH_BLOCKAGE_THRESHOLD_PCT", "8")),
            "HYDROVISION_TRASH_RACK_MODEL_PATH",
        )
        self.draft_blockage = AreaBlockageDetector(
            "draft_tube_blockage", float(os.getenv("HYDROVISION_DRAFT_BLOCKAGE_THRESHOLD_PCT", "8")),
            "HYDROVISION_DRAFT_TUBE_BLOCKAGE_MODEL_PATH",
        )
        self.gate = GatePositionVerifier()
        self.cavitation = CavitationWearDetector()
        self.thermal = ThermalHotspotDetector()

    def analyze(
        self,
        *,
        image: np.ndarray,
        media_id: int,
        content_hash: str,
        asset_id: str,
        sensor_id: str,
    ) -> list[dict]:
        gate_position = self.store.latest_gate_position() if asset_id == "intake_gate" else None
        signature = f"{content_hash}:{asset_id}:{sensor_id}:{gate_position}"
        cache_key = hashlib.sha256(signature.encode()).hexdigest()
        cached = self.store.cached_detection_events(cache_key)
        if cached:
            return cached

        events: list[SiteDetection]
        if asset_id in {"turbine_1", "turbine_2"}:
            events = [self.cavitation.predict(image)]
        elif asset_id == "intake_gate":
            events = [self.intake_blockage.predict(image), self.gate.predict(image, gate_position)]
        elif asset_id == "penstock_valve":
            events = self._legacy_events(image)
        elif asset_id == "draft_tube":
            damage = next(item for item in self._legacy_events(image) if item.detection_type == "corrosion")
            events = [damage, self.draft_blockage.predict(image)]
        elif asset_id == "main_transformer" and sensor_id.endswith("_thermal"):
            latest = self.store.latest_detection_event(asset_id, "thermal_hotspot")
            persisted_count = 0
            if latest and latest["measurement"].get("status") in {"pending_persistence", "hotspot"}:
                persisted_count = int(latest["measurement"].get("consecutive_hot_frames", 0))
            events = [self.thermal.predict(image, asset_id, persisted_count)]
        else:
            events = [self._healthy_camera_event()]
        return self.store.insert_detection_events(
            asset_id=asset_id,
            sensor_id=sensor_id,
            media_id=media_id,
            cache_key=cache_key,
            events=[event.row() for event in events],
        )

    def _legacy_events(self, image: np.ndarray) -> list[SiteDetection]:
        predictions = self.existing.predict_batch([image])[0]
        events: list[SiteDetection] = []
        for source_type, event_type in (("leak", "oil_leak"), ("corrosion", "corrosion")):
            matches = [item for item in predictions if item.defect_type == source_type]
            best = max(matches, key=lambda item: item.confidence) if matches else None
            events.append(SiteDetection(
                detection_type=event_type,
                defect_present=best is not None,
                severity=("critical" if best and (best.confidence >= 0.86 or best.affected_area_pct >= 12)
                          else "warning" if best and (best.confidence >= 0.66 or best.affected_area_pct >= 4)
                          else "observation" if best else "normal"),
                confidence=best.confidence if best else 0.9,
                bbox=best.bbox if best else None,
                measurement={
                    "affected_area_pct": best.affected_area_pct if best else 0.0,
                    "status": "detected" if best else "healthy",
                },
                engine=self.existing.engine_name,
            ))
        return events

    @staticmethod
    def _healthy_camera_event() -> SiteDetection:
        return SiteDetection(
            detection_type="thermal_hotspot",
            defect_present=False,
            severity="normal",
            confidence=None,
            bbox=None,
            measurement={"status": "unavailable", "reason": "thermal sensor required"},
            engine="sensor-routing-rule",
        )
