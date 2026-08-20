-- Phase 3: site-specific detector registry and event measurements.
CREATE TABLE IF NOT EXISTS asset (
  asset_id    TEXT PRIMARY KEY,
  name        TEXT NOT NULL,
  asset_type  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sensor (
  sensor_id   TEXT PRIMARY KEY,
  asset_id    TEXT NOT NULL REFERENCES asset(asset_id),
  sensor_type TEXT NOT NULL CHECK (sensor_type IN ('camera', 'thermal_camera'))
);

CREATE TABLE IF NOT EXISTS training_dataset (
  dataset_id    BIGSERIAL PRIMARY KEY,
  dataset_key   TEXT NOT NULL UNIQUE,
  name          TEXT NOT NULL,
  detection_type TEXT NOT NULL,
  source_url    TEXT NOT NULL,
  license       TEXT NOT NULL,
  modality      TEXT NOT NULL,
  notes         TEXT
);

CREATE TABLE IF NOT EXISTS training_image (
  image_id      BIGSERIAL PRIMARY KEY,
  dataset_id    BIGINT NOT NULL REFERENCES training_dataset(dataset_id),
  source_ref    TEXT NOT NULL,
  content_hash  TEXT NOT NULL UNIQUE,
  split         TEXT NOT NULL,
  synthetic     BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS detection_event (
  event_id       BIGSERIAL PRIMARY KEY,
  ts             TIMESTAMPTZ NOT NULL,
  asset_id       TEXT NOT NULL REFERENCES asset(asset_id),
  sensor_id      TEXT NOT NULL REFERENCES sensor(sensor_id),
  media_id       BIGINT REFERENCES media(id) ON DELETE SET NULL,
  detection_type TEXT NOT NULL,
  defect_present BOOLEAN NOT NULL,
  severity       TEXT NOT NULL,
  confidence     NUMERIC,
  bbox_json      JSONB,
  measurement    JSONB NOT NULL,
  engine         TEXT NOT NULL,
  cache_key      TEXT NOT NULL,
  created_at     TIMESTAMPTZ NOT NULL,
  UNIQUE (asset_id, sensor_id, detection_type, cache_key)
);

CREATE INDEX IF NOT EXISTS detection_event_asset_ts_idx
  ON detection_event(asset_id, ts DESC);
