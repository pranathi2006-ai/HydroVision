CREATE TABLE IF NOT EXISTS turbine_performance_curve (
  point_id   BIGSERIAL PRIMARY KEY,
  unit_id    TEXT,
  flow_m3s   NUMERIC,
  head_m     NUMERIC,
  efficiency NUMERIC
);

CREATE TABLE IF NOT EXISTS gate_flow_curve (
  point_id      BIGSERIAL PRIMARY KEY,
  gate_position NUMERIC,
  head_m        NUMERIC,
  flow_m3s      NUMERIC
);

CREATE TABLE IF NOT EXISTS hydraulic_loss_baseline (
  point_id  BIGSERIAL PRIMARY KEY,
  flow_m3s  NUMERIC,
  loss_m    NUMERIC
);
