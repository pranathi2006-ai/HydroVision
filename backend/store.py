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

SITE_ASSETS = [
    ("turbine_1", "Turbine 1", "turbine"),
    ("turbine_2", "Turbine 2", "turbine"),
    ("penstock_valve", "Penstock valve", "penstock"),
    ("main_transformer", "Main transformer", "transformer"),
    ("intake_gate", "Intake gate", "intake"),
    ("draft_tube", "Draft tube", "draft_tube"),
]

ASSET_CRITICALITY = {
    "turbine_1": 1.0,
    "turbine_2": 1.0,
    "penstock_valve": 0.9,
    "main_transformer": 0.85,
    "intake_gate": 0.85,
    "draft_tube": 0.75,
}

SITE_SENSORS = [
    ("turbine_1_camera", "turbine_1", "camera"),
    ("turbine_2_camera", "turbine_2", "camera"),
    ("penstock_valve_camera", "penstock_valve", "camera"),
    ("main_transformer_camera", "main_transformer", "camera"),
    ("main_transformer_thermal", "main_transformer", "thermal_camera"),
    ("intake_gate_camera", "intake_gate", "camera"),
    ("draft_tube_camera", "draft_tube", "camera"),
]

SITE_LAYOUT = {
    "turbine_1": {"zone": "Powerhouse 01", "x": 24, "y": 61},
    "turbine_2": {"zone": "Powerhouse 02", "x": 51, "y": 61},
    "main_transformer": {"zone": "Switchyard", "x": 78, "y": 31},
    "intake_gate": {"zone": "Upper intake", "x": 15, "y": 23},
    "penstock_valve": {"zone": "Valve gallery", "x": 42, "y": 28},
    "draft_tube": {"zone": "Lower gallery", "x": 71, "y": 75},
}

RECOMMENDED_ACTIONS = [
    ("trash_rack_blockage", "Schedule rack cleaning; re-check generation gap after cleaning."),
    ("gate_position_mismatch", "Inspect gate actuator and position calibration; verify commanded versus visual position."),
    ("oil_leak", "Isolate and inspect the leak source; repair seals and confirm flow recovery."),
    ("corrosion", "Inspect the affected surface, remove corrosion, and assess remaining thickness."),
    ("cavitation_wear", "Schedule turbine inspection and inspect runner surfaces for cavitation pitting."),
    ("draft_tube_blockage", "Inspect and clear the draft-tube obstruction; verify tailrace flow afterward."),
    ("thermal_hotspot", "Inspect transformer cooling and load connections; confirm with a calibrated thermal scan."),
]

ATTRIBUTION_RULES = [
    {
        "rule_id": "trash_rack_blockage",
        "defect_type": "trash_rack_blockage",
        "formula_type": "geometric",
        "params": {
            "operation": "rack_head_loss",
            "asset_ids": ["intake_gate"],
            "rack_open_area_m2": 22.0,
            "loss_coefficient": 1.15,
            "maximum_blockage_fraction": 0.85,
        },
        "confidence": 0.9,
    },
    {
        "rule_id": "gate_position_mismatch",
        "defect_type": "gate_position_mismatch",
        "formula_type": "geometric",
        "params": {
            "operation": "visual_gate_flow",
            "asset_ids": ["intake_gate"],
        },
        "confidence": 0.88,
    },
    {
        "rule_id": "oil_leak",
        "defect_type": "oil_leak",
        "formula_type": "heuristic_map",
        "params": {
            "operation": "flow_loss_pct",
            "asset_ids": ["penstock_valve"],
            "severity_map": {"observation": 0.25, "warning": 1.0, "critical": 2.5},
            "scale_by_event_confidence": True,
        },
        "confidence": 0.52,
    },
    {
        "rule_id": "cavitation_wear",
        "defect_type": "cavitation_wear",
        "formula_type": "heuristic_map",
        "params": {
            "operation": "efficiency_loss_pct",
            "asset_ids": ["turbine_1", "turbine_2"],
            "severity_map": {"observation": 0.4, "warning": 1.5, "critical": 4.0},
            "scale_by_event_confidence": True,
        },
        "confidence": 0.55,
    },
    {
        "rule_id": "draft_tube_blockage",
        "defect_type": "draft_tube_blockage",
        "formula_type": "heuristic_map",
        "params": {
            "operation": "head_reduction_m",
            "asset_ids": ["draft_tube"],
            "severity_map": {"observation": 0.08, "warning": 0.35, "critical": 0.9},
            "scale_by_event_confidence": True,
        },
        "confidence": 0.58,
    },
    {
        "rule_id": "draft_tube_damage",
        "defect_type": "corrosion",
        "formula_type": "heuristic_map",
        "params": {
            "operation": "head_reduction_m",
            "asset_ids": ["draft_tube"],
            "severity_map": {"observation": 0.05, "warning": 0.2, "critical": 0.55},
            "scale_by_event_confidence": True,
        },
        "confidence": 0.48,
    },
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
                CREATE TABLE IF NOT EXISTS asset (
                    asset_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    asset_type TEXT NOT NULL,
                    criticality REAL NOT NULL DEFAULT 0.5,
                    last_maintenance_at TEXT
                );
                CREATE TABLE IF NOT EXISTS sensor (
                    sensor_id TEXT PRIMARY KEY,
                    asset_id TEXT NOT NULL,
                    sensor_type TEXT NOT NULL CHECK (sensor_type IN ('camera', 'thermal_camera')),
                    FOREIGN KEY (asset_id) REFERENCES asset(asset_id)
                );
                CREATE TABLE IF NOT EXISTS training_dataset (
                    dataset_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dataset_key TEXT NOT NULL UNIQUE,
                    name TEXT NOT NULL,
                    detection_type TEXT NOT NULL,
                    source_url TEXT NOT NULL,
                    license TEXT NOT NULL,
                    modality TEXT NOT NULL,
                    notes TEXT
                );
                CREATE TABLE IF NOT EXISTS training_image (
                    image_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dataset_id INTEGER NOT NULL,
                    source_ref TEXT NOT NULL,
                    content_hash TEXT NOT NULL UNIQUE,
                    split TEXT NOT NULL,
                    synthetic INTEGER NOT NULL DEFAULT 0,
                    FOREIGN KEY (dataset_id) REFERENCES training_dataset(dataset_id)
                );
                CREATE TABLE IF NOT EXISTS detection_event (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL,
                    asset_id TEXT NOT NULL,
                    sensor_id TEXT NOT NULL,
                    media_id INTEGER,
                    detection_type TEXT NOT NULL,
                    defect_present INTEGER NOT NULL,
                    severity TEXT NOT NULL,
                    confidence REAL,
                    bbox_json TEXT,
                    measurement TEXT NOT NULL,
                    engine TEXT NOT NULL,
                    cache_key TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (asset_id) REFERENCES asset(asset_id),
                    FOREIGN KEY (sensor_id) REFERENCES sensor(sensor_id),
                    FOREIGN KEY (media_id) REFERENCES media(id) ON DELETE SET NULL,
                    UNIQUE (asset_id, sensor_id, detection_type, cache_key)
                );
                CREATE INDEX IF NOT EXISTS detection_event_asset_ts_idx
                    ON detection_event(asset_id, ts DESC);
                CREATE TABLE IF NOT EXISTS attribution_rule_config (
                    rule_id TEXT PRIMARY KEY,
                    defect_type TEXT NOT NULL,
                    formula_type TEXT NOT NULL CHECK (formula_type IN ('geometric','heuristic_map')),
                    params TEXT NOT NULL,
                    confidence REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS correlation_model_version (
                    model_id TEXT PRIMARY KEY,
                    purpose TEXT NOT NULL DEFAULT 'event_correlation',
                    model_type TEXT,
                    status TEXT NOT NULL DEFAULT 'shadow',
                    created_at TEXT NOT NULL,
                    trained_at TEXT,
                    training_rows INTEGER,
                    validation_rows INTEGER,
                    metrics_json TEXT,
                    comparison_metrics_json TEXT,
                    artifact_json TEXT,
                    defect_counts_json TEXT,
                    shadow_started_at TEXT,
                    approved_by TEXT,
                    approved_at TEXT,
                    approval_notes TEXT,
                    supersedes_model_id TEXT,
                    FOREIGN KEY (supersedes_model_id) REFERENCES correlation_model_version(model_id)
                );
                CREATE TABLE IF NOT EXISTS loss_attribution (
                    attribution_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    reading_id INTEGER NOT NULL,
                    asset_id TEXT NOT NULL,
                    event_id INTEGER NOT NULL,
                    estimated_loss_mw REAL NOT NULL,
                    confidence REAL NOT NULL,
                    method TEXT NOT NULL,
                    model_id TEXT,
                    rule_estimate_mw REAL,
                    rule_confidence REAL,
                    shadow_estimate_mw REAL,
                    shadow_probability REAL,
                    shadow_model_id TEXT,
                    model_explanation TEXT,
                    shadow_explanation TEXT,
                    FOREIGN KEY (reading_id) REFERENCES performance_reading(reading_id),
                    FOREIGN KEY (asset_id) REFERENCES asset(asset_id),
                    FOREIGN KEY (event_id) REFERENCES detection_event(event_id),
                    FOREIGN KEY (model_id) REFERENCES correlation_model_version(model_id),
                    FOREIGN KEY (shadow_model_id) REFERENCES correlation_model_version(model_id),
                    UNIQUE (reading_id, event_id)
                );
                CREATE TABLE IF NOT EXISTS attribution_run (
                    reading_id INTEGER PRIMARY KEY,
                    status TEXT NOT NULL CHECK (status IN ('attributed','unexplained')),
                    threshold_pct REAL NOT NULL,
                    processed_at TEXT NOT NULL,
                    FOREIGN KEY (reading_id) REFERENCES performance_reading(reading_id)
                );
                CREATE INDEX IF NOT EXISTS loss_attribution_reading_rank_idx
                    ON loss_attribution(reading_id, estimated_loss_mw DESC);
                CREATE TABLE IF NOT EXISTS recommended_action (
                    defect_type TEXT PRIMARY KEY,
                    default_recommendation TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS attribution_feedback (
                    feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    attribution_id INTEGER NOT NULL UNIQUE,
                    confirmed INTEGER NOT NULL,
                    notes TEXT,
                    confirmed_by TEXT NOT NULL,
                    confirmed_at TEXT NOT NULL,
                    FOREIGN KEY (attribution_id) REFERENCES loss_attribution(attribution_id)
                );
                CREATE INDEX IF NOT EXISTS attribution_feedback_confirmed_at_idx
                    ON attribution_feedback(confirmed_at DESC);
                """
            )
            media_columns = {row["name"] for row in db.execute("PRAGMA table_info(media)")}
            if "cleared_at" not in media_columns:
                db.execute("ALTER TABLE media ADD COLUMN cleared_at TEXT")
            asset_columns = {row["name"] for row in db.execute("PRAGMA table_info(asset)")}
            criticality_added = "criticality" not in asset_columns
            if criticality_added:
                db.execute("ALTER TABLE asset ADD COLUMN criticality REAL NOT NULL DEFAULT 0.5")
            if "last_maintenance_at" not in asset_columns:
                db.execute("ALTER TABLE asset ADD COLUMN last_maintenance_at TEXT")
            model_columns = {row["name"] for row in db.execute(
                "PRAGMA table_info(correlation_model_version)"
            )}
            model_column_definitions = {
                "purpose": "TEXT NOT NULL DEFAULT 'event_correlation'",
                "model_type": "TEXT",
                "status": "TEXT NOT NULL DEFAULT 'shadow'",
                "created_at": "TEXT",
                "trained_at": "TEXT",
                "training_rows": "INTEGER",
                "validation_rows": "INTEGER",
                "metrics_json": "TEXT",
                "comparison_metrics_json": "TEXT",
                "artifact_json": "TEXT",
                "defect_counts_json": "TEXT",
                "shadow_started_at": "TEXT",
                "approved_by": "TEXT",
                "approved_at": "TEXT",
                "approval_notes": "TEXT",
                "supersedes_model_id": "TEXT",
            }
            for column, definition in model_column_definitions.items():
                if column not in model_columns:
                    db.execute(
                        f"ALTER TABLE correlation_model_version ADD COLUMN {column} {definition}"
                    )
            attribution_columns = {row["name"] for row in db.execute(
                "PRAGMA table_info(loss_attribution)"
            )}
            attribution_column_definitions = {
                "model_id": "TEXT",
                "rule_estimate_mw": "REAL",
                "rule_confidence": "REAL",
                "shadow_estimate_mw": "REAL",
                "shadow_probability": "REAL",
                "shadow_model_id": "TEXT",
                "model_explanation": "TEXT",
                "shadow_explanation": "TEXT",
            }
            for column, definition in attribution_column_definitions.items():
                if column not in attribution_columns:
                    db.execute(f"ALTER TABLE loss_attribution ADD COLUMN {column} {definition}")
            db.execute(
                "CREATE INDEX IF NOT EXISTS loss_attribution_shadow_model_idx "
                "ON loss_attribution(shadow_model_id, attribution_id)"
            )
            db.executescript(
                """
                CREATE TRIGGER IF NOT EXISTS correlation_model_activation_guard
                BEFORE UPDATE OF status, approved_by, approved_at, comparison_metrics_json
                ON correlation_model_version
                WHEN NEW.purpose = 'loss_attribution' AND NEW.status = 'active'
                  AND (NEW.comparison_metrics_json IS NULL OR NEW.approved_by IS NULL OR NEW.approved_at IS NULL)
                BEGIN
                  SELECT RAISE(ABORT, 'loss-attribution model activation requires comparison metrics and explicit approval');
                END;
                CREATE TRIGGER IF NOT EXISTS correlation_model_activation_insert_guard
                BEFORE INSERT ON correlation_model_version
                WHEN NEW.purpose = 'loss_attribution' AND NEW.status = 'active'
                  AND (NEW.comparison_metrics_json IS NULL OR NEW.approved_by IS NULL OR NEW.approved_at IS NULL)
                BEGIN
                  SELECT RAISE(ABORT, 'loss-attribution model activation requires comparison metrics and explicit approval');
                END;
                """
            )
            db.executemany(
                """
                INSERT OR IGNORE INTO asset (asset_id, name, asset_type, criticality)
                VALUES (?, ?, ?, ?)
                """,
                [(*asset, ASSET_CRITICALITY[asset[0]]) for asset in SITE_ASSETS],
            )
            if criticality_added:
                db.executemany(
                    "UPDATE asset SET criticality = ? WHERE asset_id = ?",
                    [(criticality, asset_id) for asset_id, criticality in ASSET_CRITICALITY.items()],
                )
            db.executemany(
                "INSERT OR IGNORE INTO sensor (sensor_id, asset_id, sensor_type) VALUES (?, ?, ?)",
                SITE_SENSORS,
            )
            for rule in ATTRIBUTION_RULES:
                db.execute(
                    """
                    INSERT OR IGNORE INTO attribution_rule_config (
                        rule_id, defect_type, formula_type, params, confidence
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        rule["rule_id"], rule["defect_type"], rule["formula_type"],
                        json.dumps(rule["params"], sort_keys=True), rule["confidence"],
                    ),
                )
            db.executemany(
                """
                INSERT OR IGNORE INTO recommended_action (
                    defect_type, default_recommendation
                ) VALUES (?, ?)
                """,
                RECOMMENDED_ACTIONS,
            )

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

    def latest_gate_position(self) -> float | None:
        with self.connect() as db:
            row = db.execute(
                "SELECT gate_position FROM performance_reading "
                "WHERE gate_position IS NOT NULL ORDER BY ts DESC, reading_id DESC LIMIT 1"
            ).fetchone()
        return float(row["gate_position"]) if row else None

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

    def performance_reading(self, reading_id: int) -> dict | None:
        with self.connect() as db:
            row = db.execute(
                """
                SELECT reading_id, ts, headwater_level, tailwater_level,
                       gate_position, theoretical_mw, actual_mw, gap_pct
                FROM performance_reading WHERE reading_id = ?
                """,
                (reading_id,),
            ).fetchone()
        return dict(row) if row else None

    def pending_attribution_readings(self, threshold_pct: float) -> list[dict]:
        with self.connect() as db:
            rows = db.execute(
                """
                SELECT p.reading_id, p.ts, p.headwater_level, p.tailwater_level,
                       p.gate_position, p.theoretical_mw, p.actual_mw, p.gap_pct
                FROM performance_reading p
                LEFT JOIN attribution_run r ON r.reading_id = p.reading_id
                WHERE p.gap_pct > ? AND r.reading_id IS NULL
                ORDER BY p.ts, p.reading_id
                """,
                (threshold_pct,),
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

    def site_assets(self) -> list[dict]:
        with self.connect() as db:
            rows = db.execute(
                """
                SELECT a.asset_id, a.name, a.asset_type,
                       s.sensor_id, s.sensor_type
                FROM asset a JOIN sensor s ON s.asset_id = a.asset_id
                ORDER BY a.asset_id, s.sensor_type, s.sensor_id
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def sensor(self, sensor_id: str) -> dict | None:
        with self.connect() as db:
            row = db.execute("SELECT * FROM sensor WHERE sensor_id = ?", (sensor_id,)).fetchone()
        return dict(row) if row else None

    def cached_detection_events(self, cache_key: str) -> list[dict]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT * FROM detection_event WHERE cache_key = ? ORDER BY event_id",
                (cache_key,),
            ).fetchall()
        return [self._event_dict(row) for row in rows]

    def insert_detection_events(
        self,
        *,
        asset_id: str,
        sensor_id: str,
        media_id: int | None,
        cache_key: str,
        events: list[dict],
    ) -> list[dict]:
        now = utc_now()
        with self.connect() as db:
            for event in events:
                db.execute(
                    """
                    INSERT OR IGNORE INTO detection_event (
                        ts, asset_id, sensor_id, media_id, detection_type,
                        defect_present, severity, confidence, bbox_json,
                        measurement, engine, cache_key, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event.get("ts", now), asset_id, sensor_id, media_id,
                        event["detection_type"], int(event["defect_present"]),
                        event["severity"], event.get("confidence"),
                        json.dumps(event["bbox"]) if event.get("bbox") is not None else None,
                        json.dumps(event["measurement"], sort_keys=True), event["engine"],
                        cache_key, now,
                    ),
                )
        return self.cached_detection_events(cache_key)

    def detection_events(
        self,
        *,
        asset_id: str | None = None,
        since: datetime | None = None,
    ) -> list[dict]:
        clauses: list[str] = []
        values: list[str] = []
        if asset_id:
            clauses.append("asset_id = ?")
            values.append(asset_id)
        if since:
            clauses.append("ts >= ?")
            values.append(since.astimezone(timezone.utc).isoformat())
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self.connect() as db:
            rows = db.execute(
                f"SELECT * FROM detection_event {where} ORDER BY ts DESC, event_id DESC LIMIT 2000",
                values,
            ).fetchall()
        return [self._event_dict(row) for row in rows]

    def closest_prior_active_detection_events(
        self,
        reading_ts: datetime,
        *,
        window_seconds: int,
    ) -> list[dict]:
        end = reading_ts.astimezone(timezone.utc)
        start = end.timestamp() - window_seconds
        start_iso = datetime.fromtimestamp(start, timezone.utc).isoformat()
        with self.connect() as db:
            rows = db.execute(
                """
                SELECT e.*
                FROM detection_event e
                WHERE e.asset_id <> 'main_transformer'
                  AND e.ts >= ? AND e.ts <= ?
                ORDER BY e.ts DESC, e.event_id DESC
                """,
                (start_iso, end.isoformat()),
            ).fetchall()
        closest: dict[tuple[str, str], dict] = {}
        for row in rows:
            event = self._event_dict(row)
            closest.setdefault((event["asset_id"], event["detection_type"]), event)
        # A newer healthy event resolves an older defect. Filter only after
        # selecting current state; otherwise stale positive evidence would be
        # incorrectly treated as active.
        return [event for event in closest.values() if event["defect_present"]]

    def attribution_rules(self) -> list[dict]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT * FROM attribution_rule_config ORDER BY rule_id"
            ).fetchall()
        rules = []
        for row in rows:
            rule = dict(row)
            rule["params"] = json.loads(rule["params"])
            rules.append(rule)
        return rules

    def attribution_run(self, reading_id: int) -> dict | None:
        with self.connect() as db:
            row = db.execute(
                "SELECT * FROM attribution_run WHERE reading_id = ?", (reading_id,)
            ).fetchone()
        return dict(row) if row else None

    def save_attribution_run(
        self,
        reading_id: int,
        threshold_pct: float,
        contributions: list[dict],
    ) -> None:
        status = "attributed" if contributions else "unexplained"
        with self.connect() as db:
            existing = db.execute(
                "SELECT 1 FROM attribution_run WHERE reading_id = ?", (reading_id,)
            ).fetchone()
            if existing:
                return
            for contribution in contributions:
                db.execute(
                    """
                    INSERT INTO loss_attribution (
                        reading_id, asset_id, event_id, estimated_loss_mw,
                        confidence, method, model_id, rule_estimate_mw,
                        rule_confidence,
                        shadow_estimate_mw, shadow_probability, shadow_model_id,
                        model_explanation, shadow_explanation
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        reading_id, contribution["asset_id"], contribution["event_id"],
                        contribution["estimated_loss_mw"], contribution["confidence"],
                        contribution.get("method", "rule_based"), contribution.get("model_id"),
                        contribution.get("rule_estimate_mw", contribution["estimated_loss_mw"]),
                        contribution.get("rule_confidence", contribution["confidence"]),
                        contribution.get("shadow_estimate_mw"),
                        contribution.get("shadow_probability"),
                        contribution.get("shadow_model_id"),
                        json.dumps(contribution["model_explanation"], sort_keys=True)
                        if contribution.get("model_explanation") else None,
                        json.dumps(contribution["shadow_explanation"], sort_keys=True)
                        if contribution.get("shadow_explanation") else None,
                    ),
                )
            db.execute(
                """
                INSERT INTO attribution_run (reading_id, status, threshold_pct, processed_at)
                VALUES (?, ?, ?, ?)
                """,
                (reading_id, status, threshold_pct, utc_now()),
            )

    def loss_attributions(self, reading_id: int) -> list[dict]:
        with self.connect() as db:
            rows = db.execute(
                """
                SELECT l.attribution_id, l.reading_id, l.asset_id, a.name AS asset_name,
                       l.event_id, e.detection_type, l.estimated_loss_mw,
                       l.confidence, l.method, l.model_id, l.rule_estimate_mw,
                       l.rule_confidence,
                       l.shadow_estimate_mw, l.shadow_probability, l.shadow_model_id,
                       l.model_explanation, l.shadow_explanation
                FROM loss_attribution l
                JOIN asset a ON a.asset_id = l.asset_id
                JOIN detection_event e ON e.event_id = l.event_id
                WHERE l.reading_id = ?
                ORDER BY l.estimated_loss_mw DESC, l.confidence DESC, l.attribution_id
                """,
                (reading_id,),
            ).fetchall()
        result = []
        for index, row in enumerate(rows, start=1):
            item = dict(row)
            for field in ("model_explanation", "shadow_explanation"):
                item[field] = json.loads(item[field]) if item.get(field) else None
            item["rank"] = index
            result.append(item)
        return result

    def add_attribution_feedback(
        self,
        attribution_id: int,
        *,
        confirmed: bool,
        notes: str | None,
        confirmed_by: str,
    ) -> dict:
        reviewer = confirmed_by.strip()
        if not reviewer:
            raise ValueError("confirmed_by is required")
        with self.connect() as db:
            if db.execute(
                "SELECT 1 FROM loss_attribution WHERE attribution_id = ?",
                (attribution_id,),
            ).fetchone() is None:
                raise KeyError(attribution_id)
            try:
                cursor = db.execute(
                    """
                    INSERT INTO attribution_feedback (
                        attribution_id, confirmed, notes, confirmed_by, confirmed_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (attribution_id, int(confirmed), notes, reviewer, utc_now()),
                )
            except sqlite3.IntegrityError as error:
                raise ValueError("feedback already exists for this attribution") from error
            row = db.execute(
                "SELECT * FROM attribution_feedback WHERE feedback_id = ?",
                (cursor.lastrowid,),
            ).fetchone()
        result = dict(row)
        result["confirmed"] = bool(result["confirmed"])
        return result

    def confirmed_attribution_count(self) -> int:
        with self.connect() as db:
            return int(db.execute("SELECT COUNT(*) FROM attribution_feedback").fetchone()[0])

    def labeled_attribution_rows(self) -> list[dict]:
        with self.connect() as db:
            rows = db.execute(
                """
                SELECT f.feedback_id, f.confirmed, f.confirmed_at,
                       l.attribution_id, l.asset_id, l.event_id,
                       COALESCE(l.rule_estimate_mw, l.estimated_loss_mw) AS rule_estimate_mw,
                       COALESCE(l.rule_confidence, l.confidence) AS rule_confidence,
                       e.detection_type, e.severity, e.measurement,
                       a.criticality, a.last_maintenance_at,
                       p.gap_pct, p.ts AS reading_ts
                FROM attribution_feedback f
                JOIN loss_attribution l ON l.attribution_id = f.attribution_id
                JOIN detection_event e ON e.event_id = l.event_id
                JOIN asset a ON a.asset_id = l.asset_id
                JOIN performance_reading p ON p.reading_id = l.reading_id
                ORDER BY f.confirmed_at, f.feedback_id
                """
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["confirmed"] = bool(item["confirmed"])
            item["measurement"] = json.loads(item["measurement"])
            result.append(item)
        return result

    def attribution_feature_row(
        self,
        reading_id: int,
        event_id: int,
        asset_id: str,
        rule_estimate_mw: float,
        rule_confidence: float,
    ) -> dict:
        with self.connect() as db:
            row = db.execute(
                """
                SELECT ? AS rule_estimate_mw, ? AS rule_confidence,
                       ? AS asset_id, e.detection_type, e.severity, e.measurement,
                       a.criticality, a.last_maintenance_at,
                       p.gap_pct, p.ts AS reading_ts
                FROM detection_event e
                JOIN asset a ON a.asset_id = ?
                JOIN performance_reading p ON p.reading_id = ?
                WHERE e.event_id = ?
                """,
                (
                    rule_estimate_mw, rule_confidence, asset_id, asset_id,
                    reading_id, event_id,
                ),
            ).fetchone()
        if row is None:
            raise KeyError((reading_id, event_id, asset_id))
        result = dict(row)
        result["measurement"] = json.loads(result["measurement"])
        return result

    def save_loss_model_version(self, model: dict) -> None:
        with self.connect() as db:
            db.execute(
                """
                UPDATE correlation_model_version
                SET status = 'retired'
                WHERE purpose = 'loss_attribution' AND status = 'shadow'
                """
            )
            db.execute(
                """
                INSERT INTO correlation_model_version (
                    model_id, purpose, model_type, status, created_at, trained_at,
                    training_rows, validation_rows, metrics_json, artifact_json,
                    defect_counts_json, shadow_started_at, supersedes_model_id
                ) VALUES (?, 'loss_attribution', 'logistic_regression', 'shadow',
                          ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    model["model_id"], model["created_at"], model["trained_at"],
                    model["training_rows"], model["validation_rows"],
                    json.dumps(model["metrics"], sort_keys=True),
                    json.dumps(model["artifact"], sort_keys=True),
                    json.dumps(model["defect_counts"], sort_keys=True),
                    model["shadow_started_at"], model.get("supersedes_model_id"),
                ),
            )

    def loss_model(self, status: str) -> dict | None:
        if status not in {"shadow", "active"}:
            raise ValueError("status must be shadow or active")
        with self.connect() as db:
            row = db.execute(
                """
                SELECT * FROM correlation_model_version
                WHERE purpose = 'loss_attribution' AND status = ?
                ORDER BY trained_at DESC, created_at DESC LIMIT 1
                """,
                (status,),
            ).fetchone()
        return self._loss_model_dict(row) if row else None

    def loss_model_by_id(self, model_id: str) -> dict | None:
        with self.connect() as db:
            row = db.execute(
                "SELECT * FROM correlation_model_version WHERE model_id = ? AND purpose = 'loss_attribution'",
                (model_id,),
            ).fetchone()
        return self._loss_model_dict(row) if row else None

    @staticmethod
    def _loss_model_dict(row: sqlite3.Row) -> dict:
        model = dict(row)
        for column in (
            "metrics_json", "comparison_metrics_json", "artifact_json", "defect_counts_json",
        ):
            value = model.pop(column)
            model[column.removesuffix("_json")] = json.loads(value) if value else None
        return model

    def shadow_feedback_rows(self, model_id: str) -> list[dict]:
        with self.connect() as db:
            rows = db.execute(
                """
                SELECT f.confirmed, COALESCE(l.rule_confidence, l.confidence) AS rule_probability,
                       l.shadow_probability
                FROM attribution_feedback f
                JOIN loss_attribution l ON l.attribution_id = f.attribution_id
                WHERE l.shadow_model_id = ? AND l.shadow_probability IS NOT NULL
                ORDER BY f.confirmed_at, f.feedback_id
                """,
                (model_id,),
            ).fetchall()
        return [
            {
                "confirmed": bool(row["confirmed"]),
                "rule_probability": float(row["rule_probability"]),
                "shadow_probability": float(row["shadow_probability"]),
            }
            for row in rows
        ]

    def save_model_comparison(self, model_id: str, metrics: dict) -> None:
        with self.connect() as db:
            db.execute(
                """
                UPDATE correlation_model_version SET comparison_metrics_json = ?
                WHERE model_id = ? AND purpose = 'loss_attribution' AND status = 'shadow'
                """,
                (json.dumps(metrics, sort_keys=True), model_id),
            )

    def activate_loss_model(
        self,
        model_id: str,
        *,
        approved_by: str,
        approval_notes: str | None,
    ) -> None:
        with self.connect() as db:
            model = db.execute(
                """
                SELECT comparison_metrics_json FROM correlation_model_version
                WHERE model_id = ? AND purpose = 'loss_attribution' AND status = 'shadow'
                """,
                (model_id,),
            ).fetchone()
            if model is None:
                raise ValueError("model is not an eligible shadow version")
            if not model["comparison_metrics_json"]:
                raise ValueError("shadow comparison has not been recorded")
            db.execute(
                """
                UPDATE correlation_model_version SET status = 'retired'
                WHERE purpose = 'loss_attribution' AND status = 'active'
                """
            )
            db.execute(
                """
                UPDATE correlation_model_version
                SET status = 'active', approved_by = ?, approved_at = ?, approval_notes = ?
                WHERE model_id = ?
                """,
                (approved_by, utc_now(), approval_notes, model_id),
            )

    def current_dashboard(self) -> dict:
        """Return one transactionally consistent read model for both Phase 5 views."""
        with self.connect() as db:
            db.execute("BEGIN")
            reading_row = db.execute(
                """
                SELECT reading_id, ts, headwater_level, tailwater_level,
                       gate_position, theoretical_mw, actual_mw, gap_pct
                FROM performance_reading
                ORDER BY ts DESC, reading_id DESC LIMIT 1
                """
            ).fetchone()
            reading = dict(reading_row) if reading_row else None
            run = None
            attribution_rows: list[sqlite3.Row] = []
            if reading is not None:
                run = db.execute(
                    "SELECT * FROM attribution_run WHERE reading_id = ?",
                    (reading["reading_id"],),
                ).fetchone()
                attribution_rows = db.execute(
                    """
                    SELECT l.*, e.ts AS event_ts, e.sensor_id, e.detection_type,
                           e.defect_present, e.severity, e.confidence AS event_confidence,
                           e.measurement, e.bbox_json, m.stored_name
                    FROM loss_attribution l
                    JOIN detection_event e ON e.event_id = l.event_id
                    LEFT JOIN media m ON m.id = e.media_id
                    WHERE l.reading_id = ?
                    ORDER BY l.estimated_loss_mw DESC, l.confidence DESC, l.attribution_id
                    """,
                    (reading["reading_id"],),
                ).fetchall()

            latest_rows = db.execute(
                """
                SELECT e.*, m.stored_name
                FROM detection_event e
                LEFT JOIN media m ON m.id = e.media_id
                ORDER BY e.ts DESC, e.event_id DESC
                """
            ).fetchall()
            action_rows = db.execute(
                "SELECT defect_type, default_recommendation FROM recommended_action"
            ).fetchall()

        actions = {row["defect_type"]: row["default_recommendation"] for row in action_rows}
        latest_by_asset: dict[str, dict] = {}
        for row in latest_rows:
            event = self._dashboard_event_dict(row)
            latest_by_asset.setdefault(event["asset_id"], event)

        attribution_by_asset: dict[str, dict] = {}
        for rank, row in enumerate(attribution_rows, start=1):
            item = dict(row)
            item["rank"] = rank
            item["method"] = str(item["method"])
            for field in ("model_explanation", "shadow_explanation"):
                item[field] = json.loads(item[field]) if item.get(field) else None
            item["event"] = {
                "event_id": item["event_id"],
                "ts": item.pop("event_ts"),
                "asset_id": item["asset_id"],
                "sensor_id": item.pop("sensor_id"),
                "detection_type": item.pop("detection_type"),
                "defect_present": bool(item.pop("defect_present")),
                "severity": item.pop("severity"),
                "confidence": item.pop("event_confidence"),
                "measurement": json.loads(item.pop("measurement")),
                "bbox": json.loads(item.pop("bbox_json")) if item.get("bbox_json") else None,
                "thumbnail_url": f"/api/media/{item['stored_name']}" if item.get("stored_name") else None,
            }
            item.pop("bbox_json", None)
            item.pop("stored_name", None)
            attribution_by_asset[item["asset_id"]] = item

        sites = []
        for asset_id, name, asset_type in SITE_ASSETS:
            latest = latest_by_asset.get(asset_id)
            attribution = attribution_by_asset.get(asset_id)
            recommendation_type = (
                latest["detection_type"] if latest else
                attribution["event"]["detection_type"] if attribution else None
            )
            sites.append({
                "asset_id": asset_id,
                "name": name,
                "asset_type": asset_type,
                **SITE_LAYOUT[asset_id],
                "latest_event": latest,
                "attribution": attribution,
                "recommended_action": actions.get(recommendation_type) if recommendation_type else None,
            })

        sites.sort(key=lambda site: (
            -(site["attribution"]["estimated_loss_mw"] if site["attribution"] else 0),
            site["name"],
        ))
        return {
            "reading": reading,
            "attribution_status": dict(run)["status"] if run else "not_triggered",
            "sites": sites,
        }

    @staticmethod
    def _dashboard_event_dict(row: sqlite3.Row) -> dict:
        event = dict(row)
        bbox_json = event.pop("bbox_json")
        stored_name = event.pop("stored_name")
        event["defect_present"] = bool(event["defect_present"])
        event["measurement"] = json.loads(event["measurement"])
        event["bbox"] = json.loads(bbox_json) if bbox_json else None
        event["thumbnail_url"] = f"/api/media/{stored_name}" if stored_name else None
        return event

    def latest_detection_event(self, asset_id: str, detection_type: str) -> dict | None:
        with self.connect() as db:
            row = db.execute(
                """
                SELECT * FROM detection_event
                WHERE asset_id = ? AND detection_type = ?
                ORDER BY ts DESC, event_id DESC LIMIT 1
                """,
                (asset_id, detection_type),
            ).fetchone()
        return self._event_dict(row) if row else None

    @staticmethod
    def _event_dict(row: sqlite3.Row) -> dict:
        event = dict(row)
        event["defect_present"] = bool(event["defect_present"])
        event["measurement"] = json.loads(event["measurement"])
        bbox_json = event.pop("bbox_json")
        event["bbox"] = json.loads(bbox_json) if bbox_json else None
        return event

    def upsert_training_dataset(self, dataset: dict) -> int:
        with self.connect() as db:
            db.execute(
                """
                INSERT INTO training_dataset (
                    dataset_key, name, detection_type, source_url, license, modality, notes
                ) VALUES (:dataset_key, :name, :detection_type, :source_url, :license, :modality, :notes)
                ON CONFLICT(dataset_key) DO UPDATE SET
                    name=excluded.name, detection_type=excluded.detection_type,
                    source_url=excluded.source_url, license=excluded.license,
                    modality=excluded.modality, notes=excluded.notes
                """,
                dataset,
            )
            row = db.execute(
                "SELECT dataset_id FROM training_dataset WHERE dataset_key = ?",
                (dataset["dataset_key"],),
            ).fetchone()
        return int(row["dataset_id"])

    def insert_training_image(self, dataset_id: int, image: dict) -> bool:
        with self.connect() as db:
            cursor = db.execute(
                """
                INSERT OR IGNORE INTO training_image (
                    dataset_id, source_ref, content_hash, split, synthetic
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (dataset_id, image["source_ref"], image["content_hash"],
                 image["split"], int(image.get("synthetic", False))),
            )
        return cursor.rowcount == 1

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
