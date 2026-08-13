from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator


LOCATIONS = [
    {"id": "turbine-a", "name": "Turbine A", "zone": "Powerhouse 01", "x": 24, "y": 61},
    {"id": "turbine-b", "name": "Turbine B", "zone": "Powerhouse 02", "x": 51, "y": 61},
    {"id": "transformer", "name": "Main transformer", "zone": "Switchyard", "x": 78, "y": 31},
    {"id": "intake", "name": "Intake gate", "zone": "Upper intake", "x": 15, "y": 23},
    {"id": "penstock", "name": "Penstock valve", "zone": "Valve gallery", "x": 42, "y": 28},
    {"id": "draft-tube", "name": "Draft tube", "zone": "Lower gallery", "x": 71, "y": 75},
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Store:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self.connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS media (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content_hash TEXT NOT NULL UNIQUE,
                    original_name TEXT NOT NULL,
                    stored_name TEXT NOT NULL,
                    media_type TEXT NOT NULL,
                    location_id TEXT NOT NULL,
                    captured_at TEXT NOT NULL,
                    width INTEGER NOT NULL,
                    height INTEGER NOT NULL,
                    source_video TEXT,
                    sampled_second REAL,
                    inference_engine TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS findings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    media_id INTEGER NOT NULL,
                    defect_type TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    severity TEXT NOT NULL,
                    affected_area_pct REAL NOT NULL,
                    bbox_json TEXT NOT NULL,
                    review_status TEXT NOT NULL DEFAULT 'unreviewed',
                    reviewed_at TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (media_id) REFERENCES media(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS findings_created_idx ON findings(created_at DESC);
                CREATE INDEX IF NOT EXISTS media_location_idx ON media(location_id);
                """
            )

    def cached_media(self, content_hash: str) -> dict | None:
        with self.connect() as db:
            row = db.execute("SELECT * FROM media WHERE content_hash = ?", (content_hash,)).fetchone()
        return dict(row) if row else None

    def insert_media(self, **values) -> int:
        with self.connect() as db:
            cursor = db.execute(
                """
                INSERT INTO media (
                    content_hash, original_name, stored_name, media_type, location_id,
                    captured_at, width, height, source_video, sampled_second, inference_engine
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    values["content_hash"], values["original_name"], values["stored_name"],
                    values["media_type"], values["location_id"], values.get("captured_at", utc_now()),
                    values["width"], values["height"], values.get("source_video"),
                    values.get("sampled_second"), values["inference_engine"],
                ),
            )
            return int(cursor.lastrowid)

    def insert_findings(self, media_id: int, detections: list[dict]) -> None:
        with self.connect() as db:
            for item in detections:
                db.execute(
                    """
                    INSERT INTO findings (
                        media_id, defect_type, confidence, severity, affected_area_pct,
                        bbox_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        media_id, item["defect_type"], item["confidence"], item["severity"],
                        item["affected_area_pct"], json.dumps(item["bbox"]), utc_now(),
                    ),
                )

    def review(self, finding_id: int, status: str) -> bool:
        with self.connect() as db:
            cursor = db.execute(
                "UPDATE findings SET review_status = ?, reviewed_at = ? WHERE id = ?",
                (status, utc_now(), finding_id),
            )
            return cursor.rowcount == 1

    def findings(self) -> list[dict]:
        with self.connect() as db:
            rows = db.execute(
                """
                SELECT f.*, m.original_name, m.stored_name, m.location_id, m.width, m.height,
                       m.captured_at, m.source_video, m.sampled_second, m.inference_engine
                FROM findings f JOIN media m ON m.id = f.media_id
                ORDER BY f.created_at DESC, f.id DESC
                """
            ).fetchall()
        locations = {item["id"]: item for item in LOCATIONS}
        return [self._finding_dict(row, locations) for row in rows]

    @staticmethod
    def _finding_dict(row: sqlite3.Row, locations: dict) -> dict:
        data = dict(row)
        data["bbox"] = json.loads(data.pop("bbox_json"))
        data["thumbnail_url"] = f"/api/media/{data['stored_name']}"
        location = locations.get(data["location_id"], {})
        data["location_name"] = location.get("name", data["location_id"])
        data["location_zone"] = location.get("zone", "Plant")
        return data

    def snapshot(self) -> dict:
        findings = self.findings()
        priority = {"normal": 0, "observation": 1, "warning": 2, "critical": 3}
        location_rows = []
        for location in LOCATIONS:
            matching = [f for f in findings if f["location_id"] == location["id"]]
            latest = matching[0] if matching else None
            status = max((f["severity"] for f in matching if f["review_status"] != "false_positive"), key=lambda s: priority.get(s, 0), default="normal")
            location_rows.append({**location, "status": status, "finding_count": len(matching), "latest_finding": latest})

        active = [f for f in findings if f["review_status"] != "false_positive"]
        return {
            "locations": location_rows,
            "findings": findings,
            "metrics": {
                "monitored_locations": len(LOCATIONS),
                "active_findings": len(active),
                "critical_findings": sum(item["severity"] == "critical" for item in active),
                "reviewed_pct": round(sum(item["review_status"] != "unreviewed" for item in findings) / max(1, len(findings)) * 100),
            },
        }
