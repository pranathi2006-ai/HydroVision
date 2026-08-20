"use client";

import { FormEvent, useEffect, useMemo, useRef, useState } from "react";

type Severity = "normal" | "observation" | "warning" | "critical";
type Finding = {
  id: number;
  defect_type: "leak" | "corrosion";
  confidence: number;
  severity: Severity;
  affected_area_pct: number;
  bbox: [number, number, number, number];
  created_at: string;
  captured_at: string;
  location_id: string;
  location_name: string;
  location_zone: string;
  thumbnail_url: string;
  original_name: string;
  source_video?: string;
  sampled_second?: number;
  width: number;
  height: number;
};

type Location = {
  id: string;
  name: string;
  zone: string;
  x: number;
  y: number;
  status: Severity;
  finding_count: number;
  latest_finding: Finding | null;
};

type Snapshot = {
  locations: Location[];
  findings: Finding[];
  metrics: {
    monitored_locations: number;
    active_findings: number;
    critical_findings: number;
  };
};

type PerformanceReading = {
  reading_id: number;
  ts: string;
  headwater_level: number;
  tailwater_level: number;
  gate_position: number;
  theoretical_mw: number | null;
  actual_mw: number;
  gap_pct: number | null;
};

type LossAttribution = {
  attribution_id: number;
  reading_id: number;
  rank: number;
  asset_id: string;
  asset_name: string;
  event_id: number;
  detection_type: string;
  estimated_loss_mw: number;
  confidence: number;
  method: "rule_based";
};

type DetectionEvent = {
  event_id: number;
  ts: string;
  asset_id: string;
  sensor_id: string;
  detection_type: string;
  defect_present: boolean;
  severity: Severity;
  confidence: number | null;
  measurement: Record<string, unknown>;
  thumbnail_url: string | null;
};

type CurrentAttribution = LossAttribution & { event: DetectionEvent };

type DashboardSite = {
  asset_id: string;
  name: string;
  asset_type: string;
  zone: string;
  x: number;
  y: number;
  latest_event: DetectionEvent | null;
  attribution: CurrentAttribution | null;
  recommended_action: string | null;
};

type CurrentDashboard = {
  reading: PerformanceReading | null;
  attribution_status: "attributed" | "unexplained" | "not_triggered";
  sites: DashboardSite[];
  poll_interval_seconds: number;
  gap_threshold_pct: number;
  actual_mw_meter_location: "generator_terminal" | "grid_connection" | "unconfirmed";
};

type View = "twin" | "findings" | "performance";

const EMPTY: Snapshot = {
  locations: [],
  findings: [],
  metrics: { monitored_locations: 0, active_findings: 0, critical_findings: 0 },
};

// When opened from another computer, use the HydroVision host's LAN address
// instead of loopback on the viewing device. An explicit env value still wins.
const browserApiUrl = typeof window === "undefined"
  ? "http://127.0.0.1:8001"
  : `${window.location.protocol}//${window.location.hostname}:8001`;
const API = (process.env.NEXT_PUBLIC_API_URL || browserApiUrl).replace(/\/$/, "");
const configuredPerformancePollSeconds = Number(process.env.NEXT_PUBLIC_PERFORMANCE_POLL_SECONDS || 300);
const INITIAL_PERFORMANCE_POLL_SECONDS = Number.isFinite(configuredPerformancePollSeconds)
  ? Math.max(60, configuredPerformancePollSeconds)
  : 300;
function mediaUrl(path?: string) {
  return path ? `${API}${path}` : "";
}

function useCurrentDashboard() {
  const [data, setData] = useState<CurrentDashboard | null>(null);
  const [connected, setConnected] = useState<boolean | null>(null);
  const [pollSeconds, setPollSeconds] = useState(INITIAL_PERFORMANCE_POLL_SECONDS);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        const response = await fetch(`${API}/api/dashboard/current`);
        if (!response.ok) throw new Error("Dashboard service unavailable");
        const current: CurrentDashboard = await response.json();
        if (cancelled) return;
        setData(current);
        setPollSeconds(Math.max(60, current.poll_interval_seconds || INITIAL_PERFORMANCE_POLL_SECONDS));
        setConnected(true);
      } catch {
        if (!cancelled) setConnected(false);
      }
    }
    void load();
    const timer = window.setInterval(() => void load(), pollSeconds * 1000);
    return () => { cancelled = true; window.clearInterval(timer); };
  }, [pollSeconds]);

  return { data, connected, pollSeconds };
}

function relativeDate(value: string) {
  const minutes = Math.max(0, Math.round((Date.now() - new Date(value).getTime()) / 60_000));
  if (minutes < 2) return "Just now";
  if (minutes < 60) return `${minutes}m ago`;
  if (minutes < 1440) return `${Math.floor(minutes / 60)}h ago`;
  return new Intl.DateTimeFormat("en", { day: "2-digit", month: "short" }).format(new Date(value));
}

function titleCase(value: string) {
  return value.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

export function HydroVisionApp() {
  const currentDashboard = useCurrentDashboard();
  const [snapshot, setSnapshot] = useState<Snapshot>(EMPTY);
  const [activeView, setActiveView] = useState<View>("twin");
  const [performanceReadings, setPerformanceReadings] = useState<PerformanceReading[]>([]);
  const [performanceConnected, setPerformanceConnected] = useState<boolean | null>(null);
  const [performancePollSeconds, setPerformancePollSeconds] = useState(INITIAL_PERFORMANCE_POLL_SECONDS);
  const [attributionThreshold, setAttributionThreshold] = useState(5);
  const [attributionEnabled, setAttributionEnabled] = useState(true);
  const [lossAttributions, setLossAttributions] = useState<Record<number, LossAttribution[]>>({});
  const [selectedAssetId, setSelectedAssetId] = useState<string | null>(null);
  const [uploadOpen, setUploadOpen] = useState(false);
  const [connected, setConnected] = useState<boolean | null>(null);
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState("");
  const [actionNotice, setActionNotice] = useState("");
  const [clearOpen, setClearOpen] = useState(false);
  const [clearBusy, setClearBusy] = useState(false);
  const [clearError, setClearError] = useState("");
  const [query, setQuery] = useState("");
  const [severityFilter, setSeverityFilter] = useState("all");

  useEffect(() => {
    let cancelled = false;
    void fetch(`${API}/api/snapshot`)
      .then((response) => {
        if (!response.ok) throw new Error("Service unavailable");
        return response.json();
      })
      .then((data) => {
        if (cancelled) return;
        setSnapshot(data);
        setConnected(true);
      })
      .catch(() => {
        if (!cancelled) setConnected(false);
      });
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    async function loadReadings() {
      const since = new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString();
      try {
        const response = await fetch(`${API}/api/performance/readings?since=${encodeURIComponent(since)}`);
        if (!response.ok) throw new Error("Performance service unavailable");
        const readings: PerformanceReading[] = await response.json();
        setPerformanceReadings(readings);
        const sourceInterval = Number(response.headers.get("X-Poll-Interval-Seconds"));
        if (Number.isFinite(sourceInterval) && sourceInterval > 0) {
          setPerformancePollSeconds(Math.max(60, sourceInterval));
        }
        const thresholdHeader = Number(response.headers.get("X-Attribution-Threshold-Pct"));
        const threshold = Number.isFinite(thresholdHeader) ? thresholdHeader : 5;
        if (Number.isFinite(thresholdHeader)) setAttributionThreshold(thresholdHeader);
        setAttributionEnabled(response.headers.get("X-Attribution-Enabled") !== "false");
        const triggering = readings.filter((reading) => reading.gap_pct != null && reading.gap_pct > threshold);
        const ranked = await Promise.all(triggering.map(async (reading) => {
          try {
            const attributionResponse = await fetch(`${API}/api/performance/attribution?reading_id=${reading.reading_id}`);
            if (!attributionResponse.ok) return [reading.reading_id, []] as const;
            return [reading.reading_id, await attributionResponse.json() as LossAttribution[]] as const;
          } catch {
            return [reading.reading_id, []] as const;
          }
        }));
        setLossAttributions(Object.fromEntries(ranked));
        setPerformanceConnected(true);
      } catch {
        setPerformanceConnected(false);
      }
    }

    void loadReadings();
    const timer = window.setInterval(() => void loadReadings(), performancePollSeconds * 1000);
    return () => window.clearInterval(timer);
  }, [performancePollSeconds]);

  const filteredFindings = useMemo(() => {
    const term = query.trim().toLowerCase();
    return snapshot.findings.filter((finding) => {
      const matchesSeverity = severityFilter === "all" || finding.severity === severityFilter;
      const matchesText = !term || `${finding.location_name} ${finding.defect_type}`.toLowerCase().includes(term);
      return matchesSeverity && matchesText;
    });
  }, [query, severityFilter, snapshot.findings]);
  const viewCopy = activeView === "twin"
    ? ["PHASE 5 · UNIFIED OPERATIONS", "Condition and energy impact", "One current dataset rendered as a spatial twin and an energy-loss waterfall."]
    : activeView === "findings"
      ? ["INSPECTION REGISTER", "All findings", "Filter and export every detection from one source of truth."]
      : ["PHASE 4 · RULE ATTRIBUTION", "Operational performance", "Raw readings, healthy-output comparison, and evidence-linked likely contributors."];

  async function clearResults() {
    setClearBusy(true);
    setClearError("");
    try {
      const response = await fetch(`${API}/api/results`, { method: "DELETE" });
      const result = await response.json();
      if (!response.ok) throw new Error(result.detail || "Could not clear results");
      setSnapshot(result.snapshot);
      setQuery("");
      setSeverityFilter("all");
      setActionNotice(`Cleared ${result.findings_cleared} finding${result.findings_cleared === 1 ? "" : "s"}. ${result.media_cleared} image hash${result.media_cleared === 1 ? "" : "es"} retained to prevent repeat inference.`);
      setClearOpen(false);
    } catch (error) {
      setClearError(error instanceof Error ? error.message : "Could not clear results");
    } finally {
      setClearBusy(false);
    }
  }

  return (
    <main className="app-shell">
      <aside className="rail" aria-label="Primary navigation">
        <button className="brand" aria-label="HydroVision home" onClick={() => setActiveView("twin")}>
          <span className="brand-mark"><i /><i /><i /></span>
          <span>HV</span>
        </button>
        <nav>
          <button className={activeView === "twin" ? "active" : ""} onClick={() => setActiveView("twin")} aria-label="Digital twin">
            <span className="nav-icon grid-icon"><i /><i /><i /><i /></span><small>Twin</small>
          </button>
          <button className={activeView === "findings" ? "active" : ""} onClick={() => setActiveView("findings")} aria-label="Findings">
            <span className="nav-icon list-icon"><i /><i /><i /></span><small>Findings</small>
          </button>
          <button className={activeView === "performance" ? "active" : ""} onClick={() => setActiveView("performance")} aria-label="Operational readings">
            <span className="nav-icon signal-icon"><i /><i /><i /><i /></span><small>Readings</small>
          </button>
        </nav>
        <div className="rail-bottom">
          <span className={`connection-dot ${connected ? "online" : ""}`} />
          <small>{connected ? "Local" : "Offline"}</small>
        </div>
      </aside>

      <section className="workspace">
        <header className="topbar">
          <div className="site-title">
            <span>RIVERBEND</span>
            <strong>HYDRO STATION</strong>
          </div>
          <div className="topbar-actions">
            <button className="text-button" onClick={() => window.open(`${API}/api/export.csv`, "_blank")}>Export CSV</button>
            <button className="upload-button" onClick={() => setUploadOpen(true)}><span>＋</span> New inspection</button>
            <div className="avatar" aria-label="Plant engineer">AK</div>
          </div>
        </header>

        <div className="content">
          <section className="page-heading">
            <div>
              <p className="eyebrow">{viewCopy[0]}</p>
              <h1>{viewCopy[1]}</h1>
              <p>{viewCopy[2]}</p>
            </div>
            <div className="date-block">
              <span>LAST SYNC</span>
              <strong>{(activeView === "performance" ? performanceConnected : activeView === "twin" ? currentDashboard.connected : connected) ? "NOW" : "WAITING"}</strong>
            </div>
          </section>

          {activeView === "twin" && (
            <section className="metrics" aria-label="Current generation summary">
              <Metric value={formatReading(currentDashboard.data?.reading?.theoretical_mw ?? null, 2)} label="Healthy output" detail="MW · theoretical" />
              <Metric value={formatReading(currentDashboard.data?.reading?.actual_mw ?? null, 2)} label="Current output" detail="MW · actual" />
              <Metric value={formatReading(currentDashboard.data?.reading?.gap_pct ?? null, 2)} label="Generation gap" detail="% · latest reading" urgent={(currentDashboard.data?.reading?.gap_pct ?? 0) > (currentDashboard.data?.gap_threshold_pct ?? 5)} />
            </section>
          )}
          {activeView === "findings" && (
            <section className="metrics" aria-label="Plant condition summary">
              <Metric value={snapshot.metrics.monitored_locations} label="Locations monitored" detail="6 plant zones" />
              <Metric value={snapshot.metrics.active_findings} label="Active findings" detail={`${snapshot.findings.length} total recorded`} />
              <Metric value={snapshot.metrics.critical_findings} label="Critical attention" detail="Requires attention" urgent={snapshot.metrics.critical_findings > 0} />
            </section>
          )}

          {(activeView === "performance" ? performanceConnected : activeView === "twin" ? currentDashboard.connected : connected) === false && (
            <div className="offline-banner"><strong>Local service is offline.</strong> Start the backend on port 8001 to view live plant data.</div>
          )}
          {actionNotice && <div className="action-notice" role="status"><span>{actionNotice}</span><button onClick={() => setActionNotice("")} aria-label="Dismiss message">×</button></div>}

          {activeView === "twin" ? (
            <UnifiedOperationsView
              dashboard={currentDashboard.data}
              selectedAssetId={selectedAssetId}
              onSelect={setSelectedAssetId}
              pollSeconds={currentDashboard.pollSeconds}
            />
          ) : activeView === "findings" ? (
            <FindingsView
              findings={filteredFindings}
              query={query}
              setQuery={setQuery}
              severity={severityFilter}
              setSeverity={setSeverityFilter}
              onClear={() => { setClearError(""); setClearOpen(true); }}
            />
          ) : (
            <PerformanceView
              readings={performanceReadings}
              pollSeconds={performancePollSeconds}
              attributionThreshold={attributionThreshold}
              attributionEnabled={attributionEnabled}
              attributions={lossAttributions}
            />
          )}
        </div>
      </section>

      {uploadOpen && (
        <UploadDrawer
          locations={snapshot.locations}
          defaultLocation={({
            turbine_1: "turbine-a", turbine_2: "turbine-b", main_transformer: "transformer",
            intake_gate: "intake", penstock_valve: "penstock", draft_tube: "draft-tube",
          } as Record<string, string>)[selectedAssetId || ""] || "turbine-a"}
          busy={busy}
          notice={notice}
          onClose={() => { if (!busy) setUploadOpen(false); }}
          onSubmit={async (files, locationId) => {
            setBusy(true);
            setNotice("");
            const data = new FormData();
            files.forEach((file) => data.append("files", file));
            data.append("location_id", locationId);
            try {
              const response = await fetch(`${API}/api/upload`, { method: "POST", body: data });
              const result = await response.json();
              if (!response.ok) throw new Error(result.detail || "Upload failed");
              setSnapshot(result.snapshot);
              setConnected(true);
              setNotice(`${result.images_analyzed} analyzed · ${result.cached_images} from cache · ${result.findings_created} findings`);
            } catch (error) {
              setNotice(error instanceof Error ? error.message : "Upload failed");
            } finally {
              setBusy(false);
            }
          }}
        />
      )}
      {clearOpen && (
        <div className="confirm-backdrop" onMouseDown={() => { if (!clearBusy) setClearOpen(false); }}>
          <section className="confirm-dialog" role="alertdialog" aria-modal="true" aria-labelledby="clear-title" aria-describedby="clear-description" onMouseDown={(event) => event.stopPropagation()}>
            <p className="eyebrow">DESTRUCTIVE ACTION</p>
            <h2 id="clear-title">Clear all previous results?</h2>
            <p id="clear-description">This removes every finding from the dashboard and digital twin. Image hashes and local inference results stay cached so duplicate uploads are never processed twice.</p>
            {clearError && <div className="confirm-error" role="alert">{clearError}</div>}
            <div className="confirm-actions">
              <button onClick={() => setClearOpen(false)} disabled={clearBusy}>Keep results</button>
              <button className="danger-button" onClick={() => void clearResults()} disabled={clearBusy}>{clearBusy ? "Clearing…" : "Clear all results"}</button>
            </div>
          </section>
        </div>
      )}
    </main>
  );
}

function Metric({ value, label, detail, urgent = false }: { value: string | number; label: string; detail: string; urgent?: boolean }) {
  return (
    <article className={`metric ${urgent ? "urgent" : ""}`}>
      <span className="metric-value">{value}</span>
      <div><strong>{label}</strong><small>{detail}</small></div>
    </article>
  );
}

function siteCondition(site: DashboardSite): Severity {
  if (site.attribution) return "critical";
  const event = site.latest_event;
  if (!event || !event.defect_present) return "normal";
  if (event.severity === "critical" || (event.confidence ?? 0) >= 0.85) return "critical";
  return event.severity === "normal" ? "observation" : event.severity;
}

function measurementSummary(measurement: Record<string, unknown>) {
  const candidates: Array<[string, string, string]> = [
    ["blockage_pct", "Blockage", "%"],
    ["delta_t_c", "Thermal delta", " °C"],
    ["mismatch_pct_points", "Gate mismatch", " points"],
    ["visual_gate_position_pct", "Observed gate", "% open"],
    ["pitting_area_pct", "Pitting area", "%"],
    ["affected_area_pct", "Affected area", "%"],
  ];
  for (const [key, label, suffix] of candidates) {
    const value = measurement[key];
    if (typeof value === "number") return `${label}: ${value.toFixed(2)}${suffix}`;
  }
  return "No quantitative measurement recorded";
}

function UnifiedOperationsView({ dashboard, selectedAssetId, onSelect, pollSeconds }: {
  dashboard: CurrentDashboard | null;
  selectedAssetId: string | null;
  onSelect: (assetId: string | null) => void;
  pollSeconds: number;
}) {
  if (!dashboard) {
    return <div className="unified-empty"><b>Waiting for the current plant snapshot</b><span>The map and waterfall will load together.</span></div>;
  }
  const selectedSite = dashboard.sites.find((site) => site.asset_id === selectedAssetId) || null;
  return (
    <>
      <section className="unified-layout">
        <SpatialTwin sites={dashboard.sites} selectedAssetId={selectedAssetId} onSelect={onSelect} />
        <EnergyWaterfall dashboard={dashboard} selectedAssetId={selectedAssetId} onSelect={onSelect} pollSeconds={pollSeconds} />
      </section>
      {selectedSite && <SiteDetailPanel site={selectedSite} onClose={() => onSelect(null)} />}
    </>
  );
}

function SpatialTwin({ sites, selectedAssetId, onSelect }: {
  sites: DashboardSite[];
  selectedAssetId: string | null;
  onSelect: (assetId: string) => void;
}) {
  return (
    <div className="twin-card phase5-twin">
      <div className="section-bar">
        <div><span className="section-index">01</span><strong>Spatial condition</strong></div>
        <div className="legend"><span><i className="normal" /> Healthy</span><span><i className="warning" /> Inspect</span><span><i className="critical" /> Gap contributor</span></div>
      </div>
      <div className="plant-map">
        <div className="terrain-line line-a" /><div className="terrain-line line-b" /><div className="terrain-line line-c" />
        <div className="water-channel"><span>FOREBAY</span></div>
        <div className="penstock-line" />
        <div className="powerhouse"><span>POWERHOUSE</span><i /><i /><i /></div>
        <div className="switchyard"><span>SWITCHYARD</span><i /><i /><i /><i /></div>
        <div className="tailrace"><span>TAILRACE</span></div>
        {sites.map((site, index) => {
          const condition = siteCondition(site);
          return (
            <button
              key={site.asset_id}
              className={`map-marker ${condition} ${selectedAssetId === site.asset_id ? "selected" : ""} ${site.attribution ? "contributor" : ""}`}
              style={{ left: `${site.x}%`, top: `${site.y}%` }}
              onClick={() => onSelect(site.asset_id)}
              aria-pressed={selectedAssetId === site.asset_id}
              aria-label={`${site.name}: ${condition}${site.attribution ? `, ${site.attribution.estimated_loss_mw.toFixed(2)} megawatts estimated impact` : ""}`}
            >
              <b>{String(index + 1).padStart(2, "0")}</b>
              <span>{site.name}</span>
              {site.attribution && <em>{site.attribution.estimated_loss_mw.toFixed(2)} MW</em>}
            </button>
          );
        })}
        <div className="map-scale"><span>0</span><i /><span>50 M</span></div>
      </div>
      <div className="map-rank-strip">
        {sites.map((site) => (
          <button key={site.asset_id} className={selectedAssetId === site.asset_id ? "selected" : ""} onClick={() => onSelect(site.asset_id)}>
            <span className={`condition-dot ${siteCondition(site)}`} />
            <strong>{site.name}</strong>
            <small>{site.attribution ? `${site.attribution.estimated_loss_mw.toFixed(2)} MW` : "No current impact"}</small>
          </button>
        ))}
      </div>
    </div>
  );
}

function EnergyWaterfall({ dashboard, selectedAssetId, onSelect, pollSeconds }: {
  dashboard: CurrentDashboard;
  selectedAssetId: string | null;
  onSelect: (assetId: string) => void;
  pollSeconds: number;
}) {
  const reading = dashboard.reading;
  if (!reading || reading.theoretical_mw == null) {
    return <section className="waterfall-card"><div className="section-bar"><div><span className="section-index">02</span><strong>Energy waterfall</strong></div></div><div className="waterfall-empty">Waiting for a calculated performance reading.</div></section>;
  }
  const theoretical = reading.theoretical_mw;
  const actual = reading.actual_mw;
  const scale = 180 / Math.max(theoretical, 1);
  const hydraulic = dashboard.sites
    .filter((site) => site.asset_id !== "main_transformer")
    .slice()
    .sort((left, right) => (right.attribution?.estimated_loss_mw || 0) - (left.attribution?.estimated_loss_mw || 0));
  const attributedTotal = hydraulic.reduce((sum, site) => sum + (site.attribution?.estimated_loss_mw || 0), 0);
  const observedGap = Math.max(0, theoretical - actual);
  const unexplained = Math.max(0, observedGap - attributedTotal);
  const lossStages = hydraulic.map((site, index) => {
    const loss = site.attribution?.estimated_loss_mw || 0;
    const priorLoss = hydraulic.slice(0, index + 1).reduce(
      (sum, item) => sum + (item.attribution?.estimated_loss_mw || 0), 0,
    );
    return { site, loss, remaining: Math.max(0, theoretical - priorLoss) };
  });
  const expectedMechanical = Math.max(0, theoretical - attributedTotal);
  const transformer = dashboard.sites.find((site) => site.asset_id === "main_transformer")!;
  const cadence = Math.max(1, Math.round(pollSeconds / 60));

  return (
    <section className="waterfall-card">
      <div className="section-bar">
        <div><span className="section-index">02</span><strong>Energy waterfall</strong></div>
        <span className={`explanation-state ${dashboard.attribution_status}`}>{dashboard.attribution_status === "unexplained" ? "UNEXPLAINED GAP" : dashboard.attribution_status === "attributed" ? "EVIDENCE LINKED" : "NO ACTIVE GAP"}</span>
      </div>
      <div className="waterfall-summary">
        <div><span>READING</span><strong>{new Date(reading.ts).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</strong></div>
        <div><span>GAP</span><strong>{formatReading(reading.gap_pct, 2)}%</strong></div>
        <div><span>REFRESH</span><strong>{cadence} MIN</strong></div>
      </div>
      <div className="waterfall-scroll">
        <div className="waterfall-chart" role="group" aria-label="Theoretical-to-actual energy waterfall">
          <div className="waterfall-column total">
            <div className="waterfall-value">{theoretical.toFixed(2)}</div>
            <div className="waterfall-bar theoretical" style={{ height: `${theoretical * scale}px` }} />
            <strong>Theoretical</strong><small>Healthy MW</small>
          </div>
          {lossStages.map(({ site, loss, remaining: stageRemaining }) => (
            <button
              key={site.asset_id}
              className={`waterfall-column loss ${selectedAssetId === site.asset_id ? "selected" : ""} ${loss > 0 ? "active-loss" : "zero-loss"}`}
              onClick={() => onSelect(site.asset_id)}
              aria-pressed={selectedAssetId === site.asset_id}
            >
              <div className="waterfall-value">−{loss.toFixed(2)}</div>
              <div className="waterfall-bar hydraulic" style={{ height: `${Math.max(4, loss * scale)}px`, marginBottom: `${stageRemaining * scale}px` }} />
              <strong>{site.name}</strong><small>{site.asset_type === "turbine" ? "Turbine loss" : "Hydraulic loss"}</small>
            </button>
          ))}
          <div className="waterfall-column subtotal">
            <div className="waterfall-value">{expectedMechanical.toFixed(2)}</div>
            <div className="waterfall-bar mechanical" style={{ height: `${expectedMechanical * scale}px` }} />
            <strong>Expected mechanical</strong><small>After linked losses</small>
          </div>
          <button className={`waterfall-column loss electrical ${selectedAssetId === transformer.asset_id ? "selected" : ""}`} onClick={() => onSelect(transformer.asset_id)} aria-pressed={selectedAssetId === transformer.asset_id}>
            <div className="waterfall-value">0.00</div>
            <div className="waterfall-bar transformer-scope" style={{ height: "4px", marginBottom: `${expectedMechanical * scale}px` }} />
            <strong>Transformer</strong><small>{dashboard.actual_mw_meter_location === "generator_terminal" ? "Outside meter scope" : "Not attributed"}</small>
          </button>
          {unexplained > 0.005 && (
            <div className="waterfall-column loss unexplained">
              <div className="waterfall-value">−{unexplained.toFixed(2)}</div>
              <div className="waterfall-bar unexplained-bar" style={{ height: `${Math.max(6, unexplained * scale)}px`, marginBottom: `${Math.max(0, actual) * scale}px` }} />
              <strong>Unexplained</strong><small>No linked site evidence</small>
            </div>
          )}
          <div className="waterfall-column total">
            <div className="waterfall-value">{actual.toFixed(2)}</div>
            <div className="waterfall-bar actual" style={{ height: `${Math.max(0, actual) * scale}px` }} />
            <strong>Actual delivered</strong><small>Current MW</small>
          </div>
        </div>
      </div>
      <p className="waterfall-note">Segments use stored Phase 4 estimates only. Confidence remains visible in the selected-site panel; transformer condition stays a standalone risk at the generator-terminal meter boundary.</p>
    </section>
  );
}

function SiteDetailPanel({ site, onClose }: { site: DashboardSite; onClose: () => void }) {
  const event = site.latest_event;
  const attribution = site.attribution;
  return (
    <div className="site-detail-backdrop" onMouseDown={onClose}>
      <aside className="site-detail-panel" role="dialog" aria-modal="true" aria-labelledby="site-detail-title" onMouseDown={(event) => event.stopPropagation()}>
        <div className="site-detail-head">
          <div><p className="eyebrow">SYNCHRONIZED SITE DETAIL</p><h2 id="site-detail-title">{site.name}</h2><span>{site.zone}</span></div>
          <button onClick={onClose} aria-label="Close site detail">×</button>
        </div>
        <div className="site-detail-evidence">
          {event?.thumbnail_url ? <img src={mediaUrl(event.thumbnail_url)} alt={`Latest ${event.detection_type} evidence at ${site.name}`} /> : <div><b>No evidence thumbnail</b><span>Latest structured event remains available below.</span></div>}
          <span className={`severity-tag ${siteCondition(site)}`}>{siteCondition(site)}</span>
        </div>
        <dl className="site-detail-facts">
          <div><dt>Latest detection</dt><dd>{event ? titleCase(event.detection_type) : "No detection event"}</dd></div>
          <div><dt>Measurement</dt><dd>{event ? measurementSummary(event.measurement) : "—"}</dd></div>
          <div><dt>Estimated impact</dt><dd>{attribution ? `${attribution.estimated_loss_mw.toFixed(2)} MW` : "Not currently attributed"}</dd></div>
          <div><dt>Attribution confidence</dt><dd>{attribution ? `${Math.round(attribution.confidence * 100)}% · rule based` : "—"}</dd></div>
        </dl>
        <section className="recommended-action">
          <span>RECOMMENDED ACTION</span>
          <p>{site.recommended_action || "Continue routine monitoring; no active defect recommendation."}</p>
        </section>
        <p className="detail-provenance">Evidence, measurements, loss estimate, and recommendation are stored records. Selecting this site triggers no model or LLM call.</p>
      </aside>
    </div>
  );
}

function FindingsView({ findings, query, setQuery, severity, setSeverity, onClear }: {
  findings: Finding[];
  query: string;
  setQuery: (value: string) => void;
  severity: string;
  setSeverity: (value: string) => void;
  onClear: () => void;
}) {
  return (
    <section className="findings-card">
      <div className="findings-toolbar">
        <label className="search-field"><span>⌕</span><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search location or defect" /></label>
        <div className="filter-pills" aria-label="Filter by severity">
          {["all", "critical", "warning", "observation"].map((item) => <button key={item} className={severity === item ? "active" : ""} onClick={() => setSeverity(item)}>{titleCase(item)}</button>)}
        </div>
        <span className="result-count">{findings.length} RESULTS</span>
        <button className="clear-results-button" onClick={onClear}>Clear results</button>
      </div>
      <div className="table-wrap">
        <table>
          <thead><tr><th>Evidence</th><th>Location</th><th>Detection</th><th>Confidence</th><th>Severity</th><th>Captured</th></tr></thead>
          <tbody>
            {findings.map((finding) => (
              <tr key={finding.id}>
                <td><img className="table-thumb" src={mediaUrl(finding.thumbnail_url)} alt="" loading="lazy" /></td>
                <td><strong>{finding.location_name}</strong><small>{finding.location_zone}</small></td>
                <td><strong>{titleCase(finding.defect_type)}</strong><small>{finding.affected_area_pct}% affected</small></td>
                <td><span className="confidence"><i style={{ width: `${finding.confidence * 100}%` }} /></span><b>{Math.round(finding.confidence * 100)}%</b></td>
                <td><span className={`severity-tag ${finding.severity}`}>{finding.severity}</span></td>
                <td><strong>{relativeDate(finding.created_at)}</strong><small>{finding.source_video ? `Frame · ${finding.sampled_second}s` : "Image"}</small></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function formatReading(value: number | null, digits: number) {
  return value == null ? "—" : value.toFixed(digits);
}

function PerformanceView({ readings, pollSeconds, attributionThreshold, attributionEnabled, attributions }: {
  readings: PerformanceReading[];
  pollSeconds: number;
  attributionThreshold: number;
  attributionEnabled: boolean;
  attributions: Record<number, LossAttribution[]>;
}) {
  const latest = readings.at(-1);
  const newestFirst = readings.slice().reverse();
  const cadenceMinutes = Math.max(1, Math.round(pollSeconds / 60));

  return (
    <section className="performance-view">
      <div className="performance-metrics" aria-label="Latest operational reading">
        <Metric value={formatReading(latest?.actual_mw ?? null, 2)} label="Generation output" detail="MW · actual" />
        <Metric value={formatReading(latest?.headwater_level ?? null, 3)} label="Headwater level" detail="m" />
        <Metric value={formatReading(latest?.tailwater_level ?? null, 3)} label="Tailwater level" detail="m" />
        <Metric value={formatReading(latest?.gate_position ?? null, 2)} label="Gate position" detail="% open" />
      </div>
      <ActualTheoreticalChart readings={readings} />
      <div className="performance-card">
        <div className="section-bar">
          <div><span className="section-index">02</span><strong>Raw reading log · last 24 hours</strong></div>
          <span className="poll-cadence">REFRESHES EVERY {cadenceMinutes} MIN</span>
        </div>
        <div className="table-wrap">
          <table className="performance-table">
            <thead><tr><th>Timestamp</th><th>Actual</th><th>Theoretical</th><th>Gap</th><th>Likely contributors</th><th>Headwater</th><th>Tailwater</th><th>Gate position</th></tr></thead>
            <tbody>
              {newestFirst.map((reading) => (
                <tr key={reading.reading_id}>
                  <td><strong>{new Date(reading.ts).toLocaleString()}</strong><small>{relativeDate(reading.ts)}</small></td>
                  <td><b>{formatReading(reading.actual_mw, 3)}</b><small>MW</small></td>
                  <td><b>{formatReading(reading.theoretical_mw, 3)}</b><small>MW</small></td>
                  <td><b>{formatReading(reading.gap_pct, 2)}</b><small>%</small></td>
                  <td>
                    {reading.gap_pct != null && reading.gap_pct > attributionThreshold ? (
                      <AttributionRanking items={attributions[reading.reading_id]} enabled={attributionEnabled} />
                    ) : <span className="attribution-not-triggered">Below {attributionThreshold}% trigger</span>}
                  </td>
                  <td><b>{formatReading(reading.headwater_level, 3)}</b><small>m</small></td>
                  <td><b>{formatReading(reading.tailwater_level, 3)}</b><small>m</small></td>
                  <td><b>{formatReading(reading.gate_position, 2)}</b><small>% open</small></td>
                </tr>
              ))}
            </tbody>
          </table>
          {!readings.length && <div className="performance-empty"><b>Waiting for the first reading</b><span>The active source is polled on the configured minutes-scale interval.</span></div>}
        </div>
      </div>
    </section>
  );
}

function AttributionRanking({ items, enabled }: { items?: LossAttribution[]; enabled: boolean }) {
  if (!enabled) {
    return <span className="attribution-unexplained">Disabled · confirm generator-terminal meter</span>;
  }
  if (items === undefined) {
    return <span className="attribution-not-triggered">Checking recent evidence…</span>;
  }
  if (!items.length) {
    return <span className="attribution-unexplained">Unexplained · no recent active evidence</span>;
  }
  return (
    <ol className="attribution-ranking" aria-label="Ranked loss attribution">
      {items.map((item) => (
        <li key={item.attribution_id}>
          <b>{item.rank}</b>
          <span><strong>{item.asset_name}</strong><small>{item.estimated_loss_mw.toFixed(2)} MW · {Math.round(item.confidence * 100)}% confidence</small></span>
        </li>
      ))}
    </ol>
  );
}

function ActualTheoreticalChart({ readings }: { readings: PerformanceReading[] }) {
  const data = readings.filter((reading) => reading.theoretical_mw != null);
  if (data.length < 2) {
    return (
      <section className="performance-card comparison-chart-card">
        <div className="section-bar"><div><span className="section-index">01</span><strong>Actual vs theoretical output</strong></div></div>
        <div className="chart-empty">At least two calculated readings are needed for comparison.</div>
      </section>
    );
  }

  const width = 960;
  const height = 260;
  const padding = { top: 22, right: 30, bottom: 42, left: 58 };
  const times = data.map((reading) => new Date(reading.ts).getTime());
  const values = data.flatMap((reading) => [reading.actual_mw, reading.theoretical_mw as number]);
  const timeMin = Math.min(...times);
  const timeMax = Math.max(...times);
  const rawMin = Math.min(...values);
  const rawMax = Math.max(...values);
  const margin = Math.max(1, (rawMax - rawMin) * 0.12);
  const valueMin = Math.max(0, rawMin - margin);
  const valueMax = rawMax + margin;
  const x = (time: number) => padding.left + (time - timeMin) / Math.max(1, timeMax - timeMin) * (width - padding.left - padding.right);
  const y = (value: number) => padding.top + (valueMax - value) / Math.max(1, valueMax - valueMin) * (height - padding.top - padding.bottom);
  const points = (key: "actual_mw" | "theoretical_mw") => data.map((reading) => `${x(new Date(reading.ts).getTime()).toFixed(2)},${y(reading[key] as number).toFixed(2)}`).join(" ");
  const ticks = [0, 0.5, 1].map((ratio) => ({
    y: padding.top + ratio * (height - padding.top - padding.bottom),
    value: valueMax - ratio * (valueMax - valueMin),
  }));
  const latest = data.at(-1)!;
  const timeLabel = (timestamp: number) => new Intl.DateTimeFormat("en", { hour: "2-digit", minute: "2-digit", day: "2-digit", month: "short" }).format(new Date(timestamp));

  return (
    <section className="performance-card comparison-chart-card">
      <div className="section-bar">
        <div><span className="section-index">01</span><strong>Actual vs theoretical output</strong></div>
        <div className="chart-legend" aria-label="Chart legend"><span><i className="actual" />Actual {latest.actual_mw.toFixed(2)} MW</span><span><i className="theoretical" />Theoretical {(latest.theoretical_mw as number).toFixed(2)} MW</span></div>
      </div>
      <div className="comparison-chart-wrap">
        <svg className="comparison-chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-labelledby="performance-chart-title performance-chart-description">
          <title id="performance-chart-title">Actual versus theoretical generation output</title>
          <desc id="performance-chart-description">Two lines compare actual and healthy-condition theoretical megawatt output over the returned reading period.</desc>
          {ticks.map((tick) => <g key={tick.y}><line className="chart-grid" x1={padding.left} x2={width - padding.right} y1={tick.y} y2={tick.y} /><text className="chart-axis-label" x={padding.left - 10} y={tick.y + 4} textAnchor="end">{tick.value.toFixed(1)}</text></g>)}
          <text className="chart-axis-label" x={padding.left} y={height - 12}>{timeLabel(timeMin)}</text>
          <text className="chart-axis-label" x={width - padding.right} y={height - 12} textAnchor="end">{timeLabel(timeMax)}</text>
          <text className="chart-axis-label chart-unit" x="13" y={padding.top - 6}>MW</text>
          <polyline className="chart-line chart-theoretical" points={points("theoretical_mw")} />
          <polyline className="chart-line chart-actual" points={points("actual_mw")} />
        </svg>
      </div>
    </section>
  );
}

function UploadDrawer({ locations, defaultLocation, busy, notice, onClose, onSubmit }: {
  locations: Location[];
  defaultLocation: string;
  busy: boolean;
  notice: string;
  onClose: () => void;
  onSubmit: (files: File[], locationId: string) => Promise<void>;
}) {
  const [files, setFiles] = useState<File[]>([]);
  const [location, setLocation] = useState(defaultLocation);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  function submit(event: FormEvent) {
    event.preventDefault();
    if (files.length) void onSubmit(files, location);
  }

  return (
    <div className="drawer-backdrop" onMouseDown={onClose}>
      <aside className="upload-drawer" role="dialog" aria-modal="true" aria-labelledby="upload-title" onMouseDown={(event) => event.stopPropagation()}>
        <div className="drawer-head"><div><p className="eyebrow">LOCAL INFERENCE</p><h2 id="upload-title">New inspection</h2></div><button onClick={onClose} aria-label="Close">×</button></div>
        <p className="drawer-intro">Add field evidence. Images are compressed to 1024px and video is sampled every 2.5 seconds before local analysis.</p>
        <form onSubmit={submit}>
          <label className="field-label" htmlFor="location">Monitored location</label>
          <select id="location" value={location} onChange={(event) => setLocation(event.target.value)}>
            {(locations.length ? locations : [{ id: "turbine-a", name: "Turbine A", zone: "Powerhouse 01" }]).map((item) => <option key={item.id} value={item.id}>{item.name} · {item.zone}</option>)}
          </select>
          <input ref={inputRef} type="file" accept="image/jpeg,image/png,image/webp,video/mp4,video/quicktime,video/webm" multiple hidden onChange={(event) => setFiles(Array.from(event.target.files || []).slice(0, 20))} />
          <button
            type="button"
            className={`drop-zone ${dragging ? "dragging" : ""}`}
            onClick={() => inputRef.current?.click()}
            onDragOver={(event) => { event.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={(event) => { event.preventDefault(); setDragging(false); setFiles(Array.from(event.dataTransfer.files).slice(0, 20)); }}
          >
            <span className="upload-glyph">↥</span>
            <strong>Drop images or video here</strong>
            <small>or click to browse · up to 20 files / 250 MB each</small>
          </button>
          <label className="capture-button">Use device camera<input type="file" accept="image/*" capture="environment" hidden onChange={(event) => setFiles(Array.from(event.target.files || []))} /></label>
          {!!files.length && <div className="file-list"><span>{files.length} file{files.length > 1 ? "s" : ""} ready</span><small>{files.map((file) => file.name).join(" · ")}</small></div>}
          {notice && <div className="upload-notice">{notice}</div>}
          <div className="guardrails"><p><span>01</span><b>Hashed once</b><small>Duplicates use cached results</small></p><p><span>02</span><b>Sampled</b><small>No full-frame video analysis</small></p><p><span>03</span><b>Local only</b><small>No paid vision API calls</small></p></div>
          <button className="analyze-button" disabled={!files.length || busy}>{busy ? <><i /> Analyzing locally…</> : "Run inspection →"}</button>
        </form>
      </aside>
    </div>
  );
}
