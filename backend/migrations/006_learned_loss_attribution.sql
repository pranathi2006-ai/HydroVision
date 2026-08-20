-- Phase 6: feedback, interpretable learned attribution, and shadow evaluation.
-- correlation_model_version is shared with event-correlation models. The
-- CREATE is a compatibility fallback for installations where the original MVP
-- did not create the registry; it is not a second model registry.
CREATE TABLE IF NOT EXISTS correlation_model_version (
  model_id TEXT PRIMARY KEY
);

ALTER TABLE correlation_model_version
  ADD COLUMN IF NOT EXISTS purpose TEXT NOT NULL DEFAULT 'event_correlation';
ALTER TABLE correlation_model_version
  ADD COLUMN IF NOT EXISTS model_type TEXT;
ALTER TABLE correlation_model_version
  ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'shadow';
ALTER TABLE correlation_model_version
  ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ NOT NULL DEFAULT now();
ALTER TABLE correlation_model_version
  ADD COLUMN IF NOT EXISTS trained_at TIMESTAMPTZ;
ALTER TABLE correlation_model_version
  ADD COLUMN IF NOT EXISTS training_rows INTEGER;
ALTER TABLE correlation_model_version
  ADD COLUMN IF NOT EXISTS validation_rows INTEGER;
ALTER TABLE correlation_model_version
  ADD COLUMN IF NOT EXISTS metrics_json JSONB;
ALTER TABLE correlation_model_version
  ADD COLUMN IF NOT EXISTS comparison_metrics_json JSONB;
ALTER TABLE correlation_model_version
  ADD COLUMN IF NOT EXISTS artifact_json JSONB;
ALTER TABLE correlation_model_version
  ADD COLUMN IF NOT EXISTS defect_counts_json JSONB;
ALTER TABLE correlation_model_version
  ADD COLUMN IF NOT EXISTS shadow_started_at TIMESTAMPTZ;
ALTER TABLE correlation_model_version
  ADD COLUMN IF NOT EXISTS approved_by TEXT;
ALTER TABLE correlation_model_version
  ADD COLUMN IF NOT EXISTS approved_at TIMESTAMPTZ;
ALTER TABLE correlation_model_version
  ADD COLUMN IF NOT EXISTS approval_notes TEXT;
ALTER TABLE correlation_model_version
  ADD COLUMN IF NOT EXISTS supersedes_model_id TEXT REFERENCES correlation_model_version(model_id);

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint WHERE conname = 'correlation_model_version_purpose_check'
  ) THEN
    ALTER TABLE correlation_model_version ADD CONSTRAINT correlation_model_version_purpose_check
      CHECK (purpose IN ('event_correlation', 'loss_attribution'));
  END IF;
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint WHERE conname = 'correlation_model_version_status_check'
  ) THEN
    ALTER TABLE correlation_model_version ADD CONSTRAINT correlation_model_version_status_check
      CHECK (status IN ('shadow', 'active', 'retired', 'failed'));
  END IF;
END $$;

CREATE TABLE IF NOT EXISTS attribution_feedback (
  feedback_id    BIGSERIAL PRIMARY KEY,
  attribution_id BIGINT NOT NULL UNIQUE REFERENCES loss_attribution(attribution_id),
  confirmed      BOOLEAN NOT NULL,
  notes          TEXT,
  confirmed_by   TEXT NOT NULL,
  confirmed_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

ALTER TABLE asset ADD COLUMN IF NOT EXISTS criticality NUMERIC NOT NULL DEFAULT 0.5;
ALTER TABLE asset ADD COLUMN IF NOT EXISTS last_maintenance_at TIMESTAMPTZ;
UPDATE asset SET criticality = CASE asset_id
  WHEN 'turbine_1' THEN 1.0
  WHEN 'turbine_2' THEN 1.0
  WHEN 'penstock_valve' THEN 0.9
  WHEN 'main_transformer' THEN 0.85
  WHEN 'intake_gate' THEN 0.85
  WHEN 'draft_tube' THEN 0.75
  ELSE criticality
END
WHERE asset_id IN (
  'turbine_1', 'turbine_2', 'penstock_valve',
  'main_transformer', 'intake_gate', 'draft_tube'
);

ALTER TABLE loss_attribution
  ADD COLUMN IF NOT EXISTS model_id TEXT REFERENCES correlation_model_version(model_id);
ALTER TABLE loss_attribution
  ADD COLUMN IF NOT EXISTS rule_estimate_mw NUMERIC;
ALTER TABLE loss_attribution
  ADD COLUMN IF NOT EXISTS rule_confidence NUMERIC;
ALTER TABLE loss_attribution
  ADD COLUMN IF NOT EXISTS shadow_estimate_mw NUMERIC;
ALTER TABLE loss_attribution
  ADD COLUMN IF NOT EXISTS shadow_probability NUMERIC;
ALTER TABLE loss_attribution
  ADD COLUMN IF NOT EXISTS shadow_model_id TEXT REFERENCES correlation_model_version(model_id);
ALTER TABLE loss_attribution
  ADD COLUMN IF NOT EXISTS model_explanation JSONB;
ALTER TABLE loss_attribution
  ADD COLUMN IF NOT EXISTS shadow_explanation JSONB;

CREATE INDEX IF NOT EXISTS attribution_feedback_confirmed_at_idx
  ON attribution_feedback(confirmed_at DESC);
CREATE INDEX IF NOT EXISTS loss_attribution_shadow_model_idx
  ON loss_attribution(shadow_model_id, attribution_id);

-- The scheduler can create shadow versions, but the database rejects an active
-- learned model unless comparison evidence and named human approval are written
-- in the same transaction.
CREATE OR REPLACE FUNCTION guard_loss_attribution_model_activation()
RETURNS trigger AS $$
BEGIN
  IF NEW.purpose = 'loss_attribution' AND NEW.status = 'active' AND (
    NEW.comparison_metrics_json IS NULL OR NEW.approved_by IS NULL OR NEW.approved_at IS NULL
  ) THEN
    RAISE EXCEPTION 'loss-attribution model activation requires comparison metrics and explicit approval';
  END IF;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS correlation_model_activation_guard ON correlation_model_version;
CREATE TRIGGER correlation_model_activation_guard
BEFORE INSERT OR UPDATE OF status, approved_by, approved_at, comparison_metrics_json
ON correlation_model_version
FOR EACH ROW EXECUTE FUNCTION guard_loss_attribution_model_activation();
