-- Phase 5: static, zero-inference recommendations used by the detail panel.
CREATE TABLE IF NOT EXISTS recommended_action (
  defect_type            TEXT PRIMARY KEY,
  default_recommendation TEXT NOT NULL
);

INSERT INTO recommended_action (defect_type, default_recommendation)
VALUES
  ('trash_rack_blockage', 'Schedule rack cleaning; re-check generation gap after cleaning.'),
  ('gate_position_mismatch', 'Inspect gate actuator and position calibration; verify commanded versus visual position.'),
  ('oil_leak', 'Isolate and inspect the leak source; repair seals and confirm flow recovery.'),
  ('corrosion', 'Inspect the affected surface, remove corrosion, and assess remaining thickness.'),
  ('cavitation_wear', 'Schedule turbine inspection and inspect runner surfaces for cavitation pitting.'),
  ('draft_tube_blockage', 'Inspect and clear the draft-tube obstruction; verify tailrace flow afterward.'),
  ('thermal_hotspot', 'Inspect transformer cooling and load connections; confirm with a calibrated thermal scan.')
ON CONFLICT (defect_type) DO NOTHING;
