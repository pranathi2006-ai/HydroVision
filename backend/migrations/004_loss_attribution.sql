-- Phase 4: configurable, evidence-linked rule attribution.
CREATE TABLE IF NOT EXISTS attribution_rule_config (
  rule_id       TEXT PRIMARY KEY,
  defect_type   TEXT NOT NULL,
  formula_type  TEXT NOT NULL CHECK (formula_type IN ('geometric','heuristic_map')),
  params        JSONB NOT NULL,
  confidence    NUMERIC NOT NULL
);

INSERT INTO attribution_rule_config (rule_id, defect_type, formula_type, params, confidence)
VALUES
  ('trash_rack_blockage', 'trash_rack_blockage', 'geometric',
   '{"operation":"rack_head_loss","asset_ids":["intake_gate"],"rack_open_area_m2":22.0,"loss_coefficient":1.15,"maximum_blockage_fraction":0.85}'::jsonb, 0.90),
  ('gate_position_mismatch', 'gate_position_mismatch', 'geometric',
   '{"operation":"visual_gate_flow","asset_ids":["intake_gate"]}'::jsonb, 0.88),
  ('oil_leak', 'oil_leak', 'heuristic_map',
   '{"operation":"flow_loss_pct","asset_ids":["penstock_valve"],"severity_map":{"observation":0.25,"warning":1.0,"critical":2.5},"scale_by_event_confidence":true}'::jsonb, 0.52),
  ('cavitation_wear', 'cavitation_wear', 'heuristic_map',
   '{"operation":"efficiency_loss_pct","asset_ids":["turbine_1","turbine_2"],"severity_map":{"observation":0.4,"warning":1.5,"critical":4.0},"scale_by_event_confidence":true}'::jsonb, 0.55),
  ('draft_tube_blockage', 'draft_tube_blockage', 'heuristic_map',
   '{"operation":"head_reduction_m","asset_ids":["draft_tube"],"severity_map":{"observation":0.08,"warning":0.35,"critical":0.9},"scale_by_event_confidence":true}'::jsonb, 0.58),
  ('draft_tube_damage', 'corrosion', 'heuristic_map',
   '{"operation":"head_reduction_m","asset_ids":["draft_tube"],"severity_map":{"observation":0.05,"warning":0.2,"critical":0.55},"scale_by_event_confidence":true}'::jsonb, 0.48)
ON CONFLICT (rule_id) DO NOTHING;

-- This table is defined in the target platform schema; CREATE IF NOT EXISTS
-- keeps standalone Phase 1-3 installations upgradeable.
CREATE TABLE IF NOT EXISTS loss_attribution (
  attribution_id   BIGSERIAL PRIMARY KEY,
  reading_id       BIGINT NOT NULL REFERENCES performance_reading(reading_id),
  asset_id         TEXT NOT NULL REFERENCES asset(asset_id),
  event_id         BIGINT NOT NULL REFERENCES detection_event(event_id),
  estimated_loss_mw NUMERIC NOT NULL,
  confidence       NUMERIC NOT NULL,
  method           TEXT NOT NULL,
  UNIQUE (reading_id, event_id)
);

-- Records both successful and unexplained runs so triggering readings with no
-- evidence are deduplicated without inventing an attribution row.
CREATE TABLE IF NOT EXISTS attribution_run (
  reading_id    BIGINT PRIMARY KEY REFERENCES performance_reading(reading_id),
  status        TEXT NOT NULL CHECK (status IN ('attributed','unexplained')),
  threshold_pct NUMERIC NOT NULL,
  processed_at  TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS loss_attribution_reading_rank_idx
  ON loss_attribution(reading_id, estimated_loss_mw DESC);
