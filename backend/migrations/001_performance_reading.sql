CREATE TABLE IF NOT EXISTS performance_reading (
  reading_id       BIGSERIAL PRIMARY KEY,
  ts               TIMESTAMPTZ NOT NULL,
  headwater_level  NUMERIC,
  tailwater_level  NUMERIC,
  gate_position    NUMERIC,
  theoretical_mw   NUMERIC,
  actual_mw        NUMERIC,
  gap_pct          NUMERIC
);
