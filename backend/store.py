from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    from .performance import PerformanceReading
    from .reference_curves import CalculatedPerformance


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
                    inference_engine TEXT NOT NULL,
                    cleared_at TEXT
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
                CREATE TABLE IF NOT EXISTS performance_reading (
                    reading_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL,
                    headwater_level REAL,
                    tailwater_level REAL,
                    gate_position REAL,
                    theoretical_mw REAL,
                    actual_mw REAL,
                    gap_pct REAL
                );
                CREATE INDEX IF NOT EXISTS performance_reading_ts_idx
                    ON performance_reading(ts);
                CREATE TABLE IF NOT EXISTS turbine_performance_curve (
                    point_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    unit_id TEXT NOT NULL,
                    flow_m3s REAL NOT NULL,
                    head_m REAL NOT NULL,
                    efficiency REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS gate_flow_curve (
                    point_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    gate_position REAL NOT NULL,
                    head_m REAL NOT NULL,
                    flow_m3s REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS hydraulic_loss_baseline (
                    point_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    flow_m3s REAL NOT NULL,
                    loss_m REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS reference_curve_dataset (
                    dataset_id INTEGER PRIMARY KEY CHECK (dataset_id = 1),
                    dataset_name TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    is_demo INTEGER NOT NULL DEFAULT 0,
                    imported_at TEXT NOT NULL
                );
                CREATE UNIQUE INDEX IF NOT EXISTS turbine_curve_point_idx
                    ON turbine_performance_curve(unit_id, flow_m3s, head_m);
                CREATE UNIQUE INDEX IF NOT EXISTS gate_curve_point_idx
                    ON gate_flow_curve(gate_position, head_m);
                CREATE UNIQUE INDEX IF NOT EXISTS loss_curve_point_idx
                    ON hydraulic_loss_baseline(flow_m3s);
                """
            )
            media_columns = {row["name"] for row in db.execute("PRAGMA table_info(media)")}
            if "cleared_at" not in media_columns:
                db.execute("ALTER TABLE media ADD COLUMN cleared_at TEXT")

    def cached_media(self, content_hash: str) -> dict | None:
        with self.connect() as db:
            row = db.execute("SELECT * FROM media WHERE content_hash = ?", (content_hash,)).fetchone()
        return dict(row) if row else None

    def restore_media(self, media_id: int) -> None:
        with self.connect() as db:
            db.execute("UPDATE media SET cleared_at = NULL WHERE id = ?", (media_id,))

    def insert_performance_reading(
        self,
        reading: "PerformanceReading",
        calculated: "CalculatedPerformance",
    ) -> int:
        assert reading.ts is not None
        with self.connect() as db:
            cursor = db.execute(
                """
                INSERT INTO performance_reading (
                    ts, headwater_level, tailwater_level, gate_position,
                    theoretical_mw, actual_mw, gap_pct
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    reading.ts.astimezone(timezone.utc).isoformat(),
                    reading.headwater_level,
                    reading.tailwater_level,
                    reading.gate_position,
                    calculated.theoretical_mw,
                    reading.actual_mw,
                    calculated.gap_pct,
                ),
            )
            return int(cursor.lastrowid)

    def update_performance_calculation(
        self,
        reading_id: int,
        calculated: "CalculatedPerformance",
    ) -> None:
        with self.connect() as db:
            db.execute(
                """
                UPDATE performance_reading
                SET theoretical_mw = ?, gap_pct = ?
                WHERE reading_id = ?
                """,
                (calculated.theoretical_mw, calculated.gap_pct, reading_id),
            )

    def performance_readings_for_calculation(self, *, only_missing: bool) -> list[dict]:
        where = "WHERE theoretical_mw IS NULL OR gap_pct IS NULL" if only_missing else ""
        with self.connect() as db:
            rows = db.execute(
                f"""
                SELECT reading_id, ts, headwater_level, tailwater_level,
                       gate_position, theoretical_mw, actual_mw, gap_pct
                FROM performance_reading
                {where}
                ORDER BY ts ASC, reading_id ASC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def latest_performance_timestamp(self) -> datetime | None:
        with self.connect() as db:
            row = db.execute(
                "SELECT ts FROM performance_reading ORDER BY ts DESC, reading_id DESC LIMIT 1"
            ).fetchone()
        return datetime.fromisoformat(row["ts"]).astimezone(timezone.utc) if row else None

    def performance_readings_since(self, since: datetime) -> list[dict]:
        since_utc = since.astimezone(timezone.utc).isoformat()
        with self.connect() as db:
            rows = db.execute(
                """
                SELECT reading_id, ts, headwater_level, tailwater_level,
                       gate_position, theoretical_mw, actual_mw, gap_pct
                FROM performance_reading
                WHERE ts >= ?
                ORDER BY ts ASC, reading_id ASC
                """,
                (since_utc,),
            ).fetchall()
        return [dict(row) for row in rows]

    def reference_curve_dataset(self) -> dict | None:
        with self.connect() as db:
            row = db.execute("SELECT * FROM reference_curve_dataset WHERE dataset_id = 1").fetchone()
        return dict(row) if row else None

    def reference_curve_rows(self) -> dict[str, list[dict]]:
        with self.connect() as db:
            turbine = db.execute(
                "SELECT unit_id, flow_m3s, head_m, efficiency FROM turbine_performance_curve"
            ).fetchall()
            gate = db.execute(
                "SELECT gate_position, head_m, flow_m3s FROM gate_flow_curve"
            ).fetchall()
            loss = db.execute(
                "SELECT flow_m3s, loss_m FROM hydraulic_loss_baseline ORDER BY flow_m3s"
            ).fetchall()
        return {
            "turbine": [dict(row) for row in turbine],
            "gate": [dict(row) for row in gate],
            "loss": [dict(row) for row in loss],
        }

    def replace_reference_curves(
        self,
        turbine: list[dict],
        gate: list[dict],
        loss: list[dict],
        *,
        dataset_name: str,
        source_path: str,
        is_demo: bool,
    ) -> None:
        with self.connect() as db:
            db.execute("DELETE FROM turbine_performance_curve")
            db.execute("DELETE FROM gate_flow_curve")
            db.execute("DELETE FROM hydraulic_loss_baseline")
            db.executemany(
                """
                INSERT INTO turbine_performance_curve (unit_id, flow_m3s, head_m, efficiency)
                VALUES (:unit_id, :flow_m3s, :head_m, :efficiency)
                """,
                turbine,
            )
            db.executemany(
                """
                INSERT INTO gate_flow_curve (gate_position, head_m, flow_m3s)
                VALUES (:gate_position, :head_m, :flow_m3s)
                """,
                gate,
            )
            db.executemany(
                """
                INSERT INTO hydraulic_loss_baseline (flow_m3s, loss_m)
                VALUES (:flow_m3s, :loss_m)
                """,
                loss,
            )
            db.execute(
                """
                INSERT INTO reference_curve_dataset (
                    dataset_id, dataset_name, source_path, is_demo, imported_at
                ) VALUES (1, ?, ?, ?, ?)
                ON CONFLICT(dataset_id) DO UPDATE SET
                    dataset_name = excluded.dataset_name,
                    source_path = excluded.source_path,
                    is_demo = excluded.is_demo,
                    imported_at = excluded.imported_at
                """,
                (dataset_name, source_path, int(is_demo), utc_now()),
            )

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

    def clear_all(self) -> dict:
        """Hide every active result while retaining the no-reprocessing cache."""
        with self.connect() as db:
            finding_count = int(db.execute(
                "SELECT COUNT(*) FROM findings f JOIN media m ON m.id = f.media_id WHERE m.cleared_at IS NULL"
            ).fetchone()[0])
            media_count = int(db.execute("SELECT COUNT(*) FROM media WHERE cleared_at IS NULL").fetchone()[0])
            db.execute("UPDATE media SET cleared_at = ? WHERE cleared_at IS NULL", (utc_now(),))
        return {
            "findings_cleared": finding_count,
            "media_cleared": media_count,
            "cache_retained": True,
        }

    def findings(self) -> list[dict]:
        with self.connect() as db:
            rows = db.execute(
                """
                SELECT f.*, m.original_name, m.stored_name, m.location_id, m.width, m.height,
                       m.captured_at, m.source_video, m.sampled_second, m.inference_engine
                FROM findings f JOIN media m ON m.id = f.media_id
                WHERE m.cleared_at IS NULL
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
