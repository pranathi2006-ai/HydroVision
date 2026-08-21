-- Automated, auditable before/after verification and statistical promotion.
-- work_order is a compatibility definition for installations where the action
-- workflow lives outside this repository. Its dispatch approval fields remain
-- mandatory; this phase does not add an automatic dispatch path.
CREATE TABLE IF NOT EXISTS work_order (
  work_order_id       BIGSERIAL PRIMARY KEY,
  asset_id            TEXT NOT NULL REFERENCES asset(asset_id),
  event_id            BIGINT REFERENCES detection_event(event_id),
  attribution_id      BIGINT REFERENCES loss_attribution(attribution_id),
  status              TEXT NOT NULL DEFAULT 'pending_approval'
                      CHECK (status IN ('pending_approval','approved','dispatched','closed','cancelled')),
  opened_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
  closed_at            TIMESTAMPTZ,
  dispatch_approved_by TEXT,
  dispatch_approved_at TIMESTAMPTZ,
  CHECK (event_id IS NOT NULL OR attribution_id IS NOT NULL)
);

ALTER TABLE attribution_feedback ALTER COLUMN confirmed DROP NOT NULL;
ALTER TABLE attribution_feedback
  ADD COLUMN IF NOT EXISTS verification_method TEXT;
ALTER TABLE attribution_feedback
  ADD COLUMN IF NOT EXISTS sample_size_before INTEGER;
ALTER TABLE attribution_feedback
  ADD COLUMN IF NOT EXISTS sample_size_after INTEGER;
ALTER TABLE attribution_feedback
  ADD COLUMN IF NOT EXISTS gap_before_mean NUMERIC;
ALTER TABLE attribution_feedback
  ADD COLUMN IF NOT EXISTS gap_after_mean NUMERIC;
ALTER TABLE attribution_feedback
  ADD COLUMN IF NOT EXISTS p_value NUMERIC;

UPDATE attribution_feedback
SET verification_method = 'manual'
WHERE verification_method IS NULL;

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint WHERE conname = 'attribution_feedback_verification_method_check'
  ) THEN
    ALTER TABLE attribution_feedback ADD CONSTRAINT attribution_feedback_verification_method_check
      CHECK (verification_method IN ('auto_matched_condition', 'manual'));
  END IF;
END $$;

CREATE TABLE IF NOT EXISTS attribution_verification_job (
  job_id          BIGSERIAL PRIMARY KEY,
  work_order_id   BIGINT NOT NULL UNIQUE REFERENCES work_order(work_order_id),
  attribution_id  BIGINT NOT NULL REFERENCES loss_attribution(attribution_id),
  scheduled_for   TIMESTAMPTZ NOT NULL,
  status          TEXT NOT NULL DEFAULT 'pending'
                  CHECK (status IN ('pending','retrying','completed')),
  attempts        INTEGER NOT NULL DEFAULT 0,
  outcome_reason  TEXT,
  completed_at    TIMESTAMPTZ,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS attribution_verification_due_idx
  ON attribution_verification_job(status, scheduled_for);

ALTER TABLE correlation_model_version
  ADD COLUMN IF NOT EXISTS promotion_metrics JSONB;
ALTER TABLE correlation_model_version
  ADD COLUMN IF NOT EXISTS promoted_from_model_id TEXT REFERENCES correlation_model_version(model_id);
ALTER TABLE correlation_model_version
  ADD COLUMN IF NOT EXISTS auto_promoted BOOLEAN NOT NULL DEFAULT false;

CREATE OR REPLACE FUNCTION guard_manual_work_order_dispatch()
RETURNS trigger AS $$
BEGIN
  IF NEW.status IN ('dispatched', 'closed') AND (
    NEW.dispatch_approved_by IS NULL OR NEW.dispatch_approved_at IS NULL
  ) THEN
    RAISE EXCEPTION 'manual dispatch approval is required before dispatch or closure';
  END IF;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS work_order_manual_dispatch_guard ON work_order;
CREATE TRIGGER work_order_manual_dispatch_guard
BEFORE INSERT OR UPDATE OF status, dispatch_approved_by, dispatch_approved_at
ON work_order
FOR EACH ROW EXECUTE FUNCTION guard_manual_work_order_dispatch();

CREATE OR REPLACE FUNCTION schedule_attribution_verification()
RETURNS trigger AS $$
DECLARE
  linked_attribution BIGINT;
  wait_days INTEGER := COALESCE(current_setting('hydrovision.verification_wait_days', true), '14')::INTEGER;
BEGIN
  IF NEW.status = 'closed' AND (OLD.status IS DISTINCT FROM 'closed') THEN
    linked_attribution := COALESCE(
      NEW.attribution_id,
      (SELECT attribution_id FROM loss_attribution
       WHERE event_id = NEW.event_id ORDER BY attribution_id DESC LIMIT 1)
    );
    IF linked_attribution IS NOT NULL THEN
      INSERT INTO attribution_verification_job (
        work_order_id, attribution_id, scheduled_for
      ) VALUES (
        NEW.work_order_id, linked_attribution, NEW.closed_at + make_interval(days => wait_days)
      ) ON CONFLICT (work_order_id) DO NOTHING;
    END IF;
  END IF;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS work_order_verification_schedule ON work_order;
CREATE TRIGGER work_order_verification_schedule
AFTER UPDATE OF status ON work_order
FOR EACH ROW EXECUTE FUNCTION schedule_attribution_verification();

-- Replace the Phase 6 human-promotion guard. Loss-attribution models may now
-- become active only through the audited statistical auto-promotion path.
CREATE OR REPLACE FUNCTION guard_loss_attribution_model_activation()
RETURNS trigger AS $$
BEGIN
  IF NEW.purpose = 'loss_attribution' AND NEW.status = 'active' AND (
    NEW.auto_promoted IS NOT TRUE OR NEW.promotion_metrics IS NULL
  ) THEN
    RAISE EXCEPTION 'loss-attribution activation requires audited statistical auto-promotion';
  END IF;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;
