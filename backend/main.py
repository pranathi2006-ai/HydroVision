from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from fastapi import FastAPI, File, Form, HTTPException, Query, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from PIL import Image, ImageDraw, ImageOps
from pydantic import BaseModel

from .attribution import AttributionService, AttributionSettings
from .detector import Detection, LocalDetector
from .learned_attribution import (
    LearnedAttributionRetrainingScheduler,
    LearnedAttributionService,
    LearnedAttributionSettings,
    LearnedAttributionTrainer,
)
from .performance import PerformanceIngestionService, PerformanceSettings, build_adapter
from .reference_curves import (
    PerformanceCalculationService,
    PerformanceCurveModel,
    import_reference_curves,
)
from .site_detectors import DEFAULT_SENSOR, LOCATION_TO_ASSET, SiteDetectionService
from .store import LOCATIONS, Store


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("HYDROVISION_DATA_DIR", ROOT / "data"))
MEDIA_DIR = DATA_DIR / "media"
MEDIA_DIR.mkdir(parents=True, exist_ok=True)

MAX_LONG_EDGE = 1024
VIDEO_SAMPLE_SECONDS = 2.5
MAX_UPLOAD_BYTES = 250 * 1024 * 1024
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
ALLOWED_VIDEO_TYPES = {"video/mp4", "video/quicktime", "video/webm", "video/x-msvideo"}

store = Store(DATA_DIR / "hydrovision.sqlite3")
phase3_registry_path = ROOT / "training" / "phase3_dataset_registry.json"
if phase3_registry_path.is_file():
    for phase3_dataset in json.loads(phase3_registry_path.read_text()):
        store.upsert_training_dataset(phase3_dataset)
detector = LocalDetector()
site_detection = SiteDetectionService(store, detector)
performance_settings = PerformanceSettings.from_env()
reference_dataset = store.reference_curve_dataset()
if reference_dataset is None:
    if performance_settings.source != "mock":
        raise RuntimeError(
            "OEM reference curves are not loaded. Run scripts/import_reference_curves.py before starting the real source."
        )
    mock_curve_path = ROOT / "reference_curves" / "mock_design"
    import_reference_curves(
        store,
        mock_curve_path,
        dataset_name="bundled mock design curves",
        is_demo=True,
    )
    reference_dataset = store.reference_curve_dataset()
if (
    performance_settings.source == "real"
    and reference_dataset
    and reference_dataset["is_demo"]
):
    raise RuntimeError(
        "RealSourceAdapter cannot run with bundled mock reference curves. Import the plant OEM/design curves first."
    )
performance_calculation = PerformanceCalculationService(
    store,
    PerformanceCurveModel(store, performance_settings.unit_id),
    nameplate_capacity_mw=performance_settings.nameplate_capacity_mw,
)
performance_backfill_summary = performance_calculation.backfill()
attribution_settings = AttributionSettings.from_env(performance_settings.source)
learned_attribution_settings = LearnedAttributionSettings.from_env()
learned_attribution_service = LearnedAttributionService(store, learned_attribution_settings)
learned_attribution_trainer = LearnedAttributionTrainer(store, learned_attribution_settings)
learned_attribution_scheduler = LearnedAttributionRetrainingScheduler(
    learned_attribution_trainer, learned_attribution_settings,
)
attribution_service = AttributionService(
    store,
    performance_calculation.curves,
    attribution_settings,
    learned_service=learned_attribution_service,
)
attribution_backfill_summary = attribution_service.backfill()
performance_ingestion = PerformanceIngestionService(
    store,
    build_adapter(performance_settings),
    performance_settings,
    performance_calculation,
    on_reading_stored=attribution_service.on_reading_stored,
)

allowed_origins = {
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://hydrovision-local-inspection.deepdrishti-8567.chatgpt.site",
}
allowed_origins.update(
    origin.strip()
    for origin in os.getenv("HYDROVISION_ALLOWED_ORIGINS", "").split(",")
    if origin.strip()
)

# Allow browser clients served from this machine over private Wi-Fi/LAN
# addresses while avoiding a blanket public-origin CORS policy.
LAN_ORIGIN_REGEX = (
    r"^http://(?:localhost|127\.0\.0\.1|10(?:\.\d{1,3}){3}|"
    r"192\.168(?:\.\d{1,3}){2}|172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2}|"
    r"169\.254(?:\.\d{1,3}){2}|[A-Za-z0-9-]+\.local)(?::\d{1,5})?$"
)

@asynccontextmanager
async def lifespan(_: FastAPI):
    performance_ingestion.start()
    learned_attribution_scheduler.start()
    try:
        yield
    finally:
        await learned_attribution_scheduler.stop()
        await performance_ingestion.stop()


app = FastAPI(title="HydroVision local inspection API", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=sorted(allowed_origins),
    allow_origin_regex=LAN_ORIGIN_REGEX,
    allow_credentials=False,
    allow_methods=["GET", "POST", "PATCH", "DELETE"],
    allow_headers=["Content-Type"],
    expose_headers=[
        "X-Poll-Interval-Seconds",
        "X-Attribution-Threshold-Pct",
        "X-Attribution-Enabled",
        "X-Attribution-Meter-Location",
    ],
)


class ReviewRequest(BaseModel):
    status: str


class AttributionFeedbackRequest(BaseModel):
    confirmed: bool
    notes: str | None = None
    confirmed_by: str


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", Path(name).name).strip("-") or "inspection"


def severity_for(item: Detection) -> str:
    if item.confidence >= 0.86 or item.affected_area_pct >= 12:
        return "critical"
    if item.confidence >= 0.66 or item.affected_area_pct >= 4:
        return "warning"
    return "observation"


def normalize_image(raw: bytes) -> tuple[bytes, np.ndarray, int, int]:
    try:
        image = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
    except Exception as error:
        raise HTTPException(status_code=400, detail="The uploaded image could not be decoded.") from error
    width, height = image.size
    scale = min(1.0, MAX_LONG_EDGE / max(width, height))
    if scale < 1:
        image = image.resize((round(width * scale), round(height * scale)), Image.Resampling.LANCZOS)
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=82, optimize=True)
    jpeg = output.getvalue()
    array = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    return jpeg, array, image.width, image.height


def detection_dict(item: Detection) -> dict:
    return {
        "defect_type": item.defect_type,
        "confidence": item.confidence,
        "bbox": list(item.bbox),
        "affected_area_pct": item.affected_area_pct,
        "severity": severity_for(item),
    }


def persist_image(
    raw: bytes,
    original_name: str,
    location_id: str,
    *,
    source_video: str | None = None,
    sampled_second: float | None = None,
) -> tuple[dict, np.ndarray | None, int | None]:
    content_hash = hashlib.sha256(raw).hexdigest()
    cached = store.cached_media(content_hash)
    if cached:
        store.restore_media(cached["id"])
        return {
            "cached": True,
            "media_id": cached["id"],
            "hash": content_hash,
            "stored_name": cached["stored_name"],
        }, None, None

    jpeg, image, width, height = normalize_image(raw)
    stored_name = f"{content_hash[:24]}.jpg"
    (MEDIA_DIR / stored_name).write_bytes(jpeg)
    media_id = store.insert_media(
        content_hash=content_hash,
        original_name=safe_name(original_name),
        stored_name=stored_name,
        media_type="video_frame" if source_video else "image",
        location_id=location_id,
        width=width,
        height=height,
        source_video=source_video,
        sampled_second=sampled_second,
        inference_engine=detector.engine_name,
    )
    return {
        "cached": False,
        "media_id": media_id,
        "hash": content_hash,
        "stored_name": stored_name,
    }, image, media_id


def seed_demo_data() -> None:
    """Create a small, local evidence set so the first-run dashboard is useful."""
    if store.findings():
        return
    samples = [
        ("turbine-a", "corrosion", "warning", 0.78, 7.4, (265, 132, 482, 311), "#994521"),
        ("transformer", "leak", "critical", 0.91, 13.8, (225, 260, 510, 382), "#263b40"),
        ("penstock", "corrosion", "observation", 0.61, 2.7, (410, 150, 526, 252), "#a6532b"),
    ]
    for index, (location, defect, severity, confidence, area, bbox, patch_color) in enumerate(samples):
        image = Image.new("RGB", (720, 460), "#9fa9a5")
        draw = ImageDraw.Draw(image)
        draw.rectangle((70, 78, 650, 392), fill="#596763", outline="#d0d6d2", width=5)
        draw.ellipse((145, 118, 430, 383), fill="#77837e", outline="#293834", width=12)
        draw.ellipse((205, 178, 370, 343), fill="#354641", outline="#a7b2ad", width=8)
        draw.rectangle(bbox, fill=patch_color)
        for offset in range(0, 130, 26):
            draw.line((95 + offset, 86, 75 + offset, 384), fill="#74817c", width=3)
        buffer = io.BytesIO()
        image.save(buffer, "JPEG", quality=88)
        result, _, media_id = persist_image(buffer.getvalue(), f"commissioning-check-{index + 1}.jpg", location)
        if result["cached"] or media_id is None:
            continue
        store.insert_findings(
            media_id,
            [{
                "defect_type": defect,
                "confidence": confidence,
                "severity": severity,
                "affected_area_pct": area,
                "bbox": list(bbox),
            }],
        )


if os.getenv("HYDROVISION_SEED_DEMO", "").lower() in {"1", "true", "yes"}:
    seed_demo_data()


def sample_video(path: Path, original_name: str, location_id: str) -> tuple[list[dict], list[tuple[np.ndarray, int]]]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise HTTPException(status_code=400, detail="The uploaded video could not be decoded.")
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if frame_count > 0 else 0
    timestamps = np.arange(0, duration, VIDEO_SAMPLE_SECONDS) if duration > 0 else np.array([0.0])
    processed: list[dict] = []
    pending: list[tuple[np.ndarray, int]] = []
    for second in timestamps:
        capture.set(cv2.CAP_PROP_POS_MSEC, float(second) * 1000)
        ok, frame = capture.read()
        if not ok:
            continue
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            continue
        result, image, media_id = persist_image(
            encoded.tobytes(),
            f"{Path(original_name).stem}-frame-{second:05.1f}.jpg",
            location_id,
            source_video=safe_name(original_name),
            sampled_second=round(float(second), 2),
        )
        processed.append(result)
        if image is not None and media_id is not None:
            pending.append((image, media_id))
    capture.release()
    return processed, pending


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ready",
        "inference": "local-only",
        "engine": detector.engine_name,
        "max_long_edge": MAX_LONG_EDGE,
        "video_sample_seconds": VIDEO_SAMPLE_SECONDS,
        "performance_source": performance_settings.source,
        "performance_poll_interval_seconds": performance_settings.effective_interval_seconds,
        "performance_unit_id": performance_settings.unit_id,
        "nameplate_capacity_mw": performance_settings.nameplate_capacity_mw,
        "reference_curve_dataset": reference_dataset["dataset_name"] if reference_dataset else None,
        "performance_backfill": performance_backfill_summary,
        "attribution": {
            "enabled": attribution_settings.enabled,
            "gap_threshold_pct": attribution_settings.threshold_pct,
            "evidence_window_seconds": attribution_settings.evidence_window_seconds,
            "actual_mw_meter_location": attribution_settings.meter_location,
            "backfill": attribution_backfill_summary,
            "transformer_excluded": True,
            "learned": {
                "confirmed_outcomes": store.confirmed_attribution_count(),
                "shadow_model_id": (
                    store.loss_model("shadow") or {}
                ).get("model_id"),
                "active_model_id": (
                    store.loss_model("active") or {}
                ).get("model_id"),
                "minimum_rows_per_defect": learned_attribution_settings.minimum_rows_per_defect,
                "scheduled_retraining_is_shadow_only": True,
            },
        },
        "phase3": {
            "assets": len({row["asset_id"] for row in store.site_assets()}),
            "sensors": len(store.site_assets()),
            "inference": "local-only",
        },
    }


@app.get("/api/performance/readings")
def performance_readings(
    response: Response,
    since: datetime = Query(..., description="Timezone-aware ISO-8601 lower bound"),
) -> list[dict]:
    if since.tzinfo is None:
        raise HTTPException(status_code=422, detail="since must include a timezone offset")
    response.headers["X-Poll-Interval-Seconds"] = str(
        performance_settings.effective_interval_seconds
    )
    response.headers["X-Attribution-Threshold-Pct"] = str(
        attribution_settings.threshold_pct
    )
    response.headers["X-Attribution-Enabled"] = str(attribution_settings.enabled).lower()
    response.headers["X-Attribution-Meter-Location"] = attribution_settings.meter_location
    return store.performance_readings_since(since.astimezone(timezone.utc))


@app.get("/api/performance/attribution")
def performance_attribution(reading_id: int = Query(..., gt=0)) -> list[dict]:
    reading = store.performance_reading(reading_id)
    if reading is None:
        raise HTTPException(status_code=404, detail="Performance reading not found.")
    attribution_service.attribute_reading(reading_id)
    return store.loss_attributions(reading_id)


@app.post("/api/loss-attribution/{attribution_id}/feedback", status_code=201)
def attribution_feedback(attribution_id: int, payload: AttributionFeedbackRequest) -> dict:
    try:
        return store.add_attribution_feedback(
            attribution_id,
            confirmed=payload.confirmed,
            notes=payload.notes,
            confirmed_by=payload.confirmed_by,
        )
    except KeyError as error:
        raise HTTPException(status_code=404, detail="Loss attribution not found.") from error
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.get("/api/dashboard/current")
def current_dashboard(response: Response) -> dict:
    response.headers["X-Poll-Interval-Seconds"] = str(
        performance_settings.effective_interval_seconds
    )
    return {
        **store.current_dashboard(),
        "poll_interval_seconds": performance_settings.effective_interval_seconds,
        "gap_threshold_pct": attribution_settings.threshold_pct,
        "actual_mw_meter_location": attribution_settings.meter_location,
    }


@app.get("/api/snapshot")
def snapshot() -> dict:
    return store.snapshot()


@app.get("/api/locations")
def locations() -> list[dict]:
    return LOCATIONS


@app.get("/api/site-assets")
def site_assets() -> list[dict]:
    return store.site_assets()


@app.get("/api/detection-events")
def detection_events(
    asset_id: str | None = Query(default=None),
    since: datetime | None = Query(default=None),
) -> list[dict]:
    if asset_id and asset_id not in DEFAULT_SENSOR:
        raise HTTPException(status_code=400, detail="Unknown asset.")
    if since and since.tzinfo is None:
        raise HTTPException(status_code=422, detail="since must include a timezone offset")
    return store.detection_events(asset_id=asset_id, since=since)


@app.post("/api/upload")
async def upload(
    files: list[UploadFile] = File(...),
    location_id: str = Form(...),
    sensor_id: str | None = Form(default=None),
) -> dict:
    if location_id not in {item["id"] for item in LOCATIONS}:
        raise HTTPException(status_code=400, detail="Unknown monitored location.")
    if not files or len(files) > 20:
        raise HTTPException(status_code=400, detail="Upload between 1 and 20 files per batch.")
    asset_id = LOCATION_TO_ASSET[location_id]
    selected_sensor = sensor_id or DEFAULT_SENSOR[asset_id]
    sensor = store.sensor(selected_sensor)
    if sensor is None or sensor["asset_id"] != asset_id:
        raise HTTPException(status_code=400, detail="Sensor is not assigned to the selected asset.")

    processed: list[dict] = []
    pending: list[tuple[np.ndarray, int]] = []
    video_frames = 0
    for incoming in files:
        content_type = (incoming.content_type or "").lower()
        if content_type not in ALLOWED_IMAGE_TYPES | ALLOWED_VIDEO_TYPES:
            raise HTTPException(status_code=415, detail=f"Unsupported file type: {content_type or 'unknown'}")
        raw = await incoming.read(MAX_UPLOAD_BYTES + 1)
        if len(raw) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Files must be 250 MB or smaller.")

        if content_type in ALLOWED_VIDEO_TYPES:
            video_hash = hashlib.sha256(raw).hexdigest()
            temp_path = DATA_DIR / f"upload-{video_hash[:20]}{Path(incoming.filename or 'video.mp4').suffix}"
            temp_path.write_bytes(raw)
            try:
                video_results, video_pending = sample_video(temp_path, incoming.filename or "video", location_id)
            finally:
                temp_path.unlink(missing_ok=True)
            processed.extend(video_results)
            pending.extend(video_pending)
            video_frames += len(video_results)
        else:
            result, image, media_id = persist_image(raw, incoming.filename or "image", location_id)
            processed.append(result)
            if image is not None and media_id is not None:
                pending.append((image, media_id))

    # All uncached images/frames are inferred together to avoid per-image setup.
    predictions = detector.predict_batch(image for image, _ in pending)
    finding_count = 0
    for (_, media_id), detections in zip(pending, predictions):
        rows = [detection_dict(item) for item in detections]
        store.insert_findings(media_id, rows)
        finding_count += len(rows)

    new_images = {media_id: image for image, media_id in pending}
    event_rows: list[dict] = []
    for item in processed:
        media_id = int(item["media_id"])
        image = new_images.get(media_id)
        if image is None:
            image = cv2.imread(str(MEDIA_DIR / item["stored_name"]))
        if image is None:
            raise HTTPException(status_code=500, detail="Stored evidence image could not be decoded.")
        event_rows.extend(site_detection.analyze(
            image=image,
            media_id=media_id,
            content_hash=item["hash"],
            asset_id=asset_id,
            sensor_id=selected_sensor,
        ))

    return {
        "files_received": len(files),
        "images_analyzed": len(pending),
        "cached_images": sum(bool(item["cached"]) for item in processed),
        "sampled_video_frames": video_frames,
        "findings_created": finding_count,
        "detection_events": event_rows,
        "site_events_returned": len(event_rows),
        "asset_id": asset_id,
        "sensor_id": selected_sensor,
        "sample_interval_seconds": VIDEO_SAMPLE_SECONDS,
        "snapshot": store.snapshot(),
    }


@app.patch("/api/findings/{finding_id}/review")
def review_finding(finding_id: int, request: ReviewRequest) -> dict:
    if request.status not in {"true_positive", "false_positive", "unreviewed"}:
        raise HTTPException(status_code=400, detail="Invalid review status.")
    if not store.review(finding_id, request.status):
        raise HTTPException(status_code=404, detail="Finding not found.")
    return store.snapshot()


@app.delete("/api/results")
def clear_results() -> dict:
    cleared = store.clear_all()
    return {
        **cleared,
        "snapshot": store.snapshot(),
    }


@app.get("/api/media/{stored_name}")
def media(stored_name: str):
    if safe_name(stored_name) != stored_name:
        raise HTTPException(status_code=400, detail="Invalid media name.")
    path = MEDIA_DIR / stored_name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Media not found.")
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "public, max-age=31536000, immutable"})


@app.get("/api/export.csv")
def export_csv():
    rows = store.findings()
    output = io.StringIO()
    columns = ["id", "created_at", "location_name", "defect_type", "confidence", "severity", "affected_area_pct", "review_status", "original_name"]
    writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=hydrovision-findings.csv"},
    )
