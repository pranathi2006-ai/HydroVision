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
  const [snapshot, setSnapshot] = useState<Snapshot>(EMPTY);
  const [activeView, setActiveView] = useState<View>("twin");
  const [performanceReadings, setPerformanceReadings] = useState<PerformanceReading[]>([]);
  const [performanceConnected, setPerformanceConnected] = useState<boolean | null>(null);
  const [performancePollSeconds, setPerformancePollSeconds] = useState(INITIAL_PERFORMANCE_POLL_SECONDS);
  const [selectedLocation, setSelectedLocation] = useState<string>("turbine-a");
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
        setPerformanceReadings(await response.json());
        const sourceInterval = Number(response.headers.get("X-Poll-Interval-Seconds"));
        if (Number.isFinite(sourceInterval) && sourceInterval > 0) {
          setPerformancePollSeconds(Math.max(60, sourceInterval));
        }
        setPerformanceConnected(true);
      } catch {
        setPerformanceConnected(false);
      }
    }

    void loadReadings();
    const timer = window.setInterval(() => void loadReadings(), performancePollSeconds * 1000);
    return () => window.clearInterval(timer);
  }, [performancePollSeconds]);

  const selected = snapshot.locations.find((item) => item.id === selectedLocation) ?? snapshot.locations[0];
  const filteredFindings = useMemo(() => {
    const term = query.trim().toLowerCase();
    return snapshot.findings.filter((finding) => {
      const matchesSeverity = severityFilter === "all" || finding.severity === severityFilter;
      const matchesText = !term || `${finding.location_name} ${finding.defect_type}`.toLowerCase().includes(term);
      return matchesSeverity && matchesText;
    });
  }, [query, severityFilter, snapshot.findings]);
  const viewCopy = activeView === "twin"
    ? ["LIVE ASSET OVERVIEW", "Plant condition", "One synchronized view of visual risk across monitored equipment."]
    : activeView === "findings"
      ? ["INSPECTION REGISTER", "All findings", "Filter and export every detection from one source of truth."]
      : ["PHASE 1 · RAW DATA", "Operational readings", "Four source readings, stored as received with no calculations or aggregation."];

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
              <strong>{(activeView === "performance" ? performanceConnected : connected) ? "NOW" : "WAITING"}</strong>
            </div>
          </section>

          {activeView !== "performance" && (
            <section className="metrics" aria-label="Plant condition summary">
              <Metric value={snapshot.metrics.monitored_locations} label="Locations monitored" detail="6 plant zones" />
              <Metric value={snapshot.metrics.active_findings} label="Active findings" detail={`${snapshot.findings.length} total recorded`} />
              <Metric value={snapshot.metrics.critical_findings} label="Critical attention" detail="Requires attention" urgent={snapshot.metrics.critical_findings > 0} />
            </section>
          )}

          {(activeView === "performance" ? performanceConnected : connected) === false && (
            <div className="offline-banner"><strong>Local service is offline.</strong> Start the backend on port 8001 to view live plant data.</div>
          )}
          {actionNotice && <div className="action-notice" role="status"><span>{actionNotice}</span><button onClick={() => setActionNotice("")} aria-label="Dismiss message">×</button></div>}

          {activeView === "twin" ? (
            <TwinView
              locations={snapshot.locations}
              findings={snapshot.findings}
              selected={selected}
              onSelect={setSelectedLocation}
              onShowFindings={() => setActiveView("findings")}
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
            <PerformanceView readings={performanceReadings} pollSeconds={performancePollSeconds} />
          )}
        </div>
      </section>

      {uploadOpen && (
        <UploadDrawer
          locations={snapshot.locations}
          defaultLocation={selected?.id || "turbine-a"}
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

function TwinView({ locations, findings, selected, onSelect, onShowFindings }: {
  locations: Location[];
  findings: Finding[];
  selected?: Location;
  onSelect: (id: string) => void;
  onShowFindings: () => void;
}) {
  return (
    <section className="twin-layout">
      <div className="twin-card">
        <div className="section-bar">
          <div><span className="section-index">01</span><strong>Digital twin</strong></div>
          <div className="legend"><span><i className="normal" /> Normal</span><span><i className="observation" /> Observe</span><span><i className="warning" /> Warning</span><span><i className="critical" /> Critical</span></div>
        </div>
        <div className="plant-map">
          <div className="terrain-line line-a" /><div className="terrain-line line-b" /><div className="terrain-line line-c" />
          <div className="water-channel"><span>FOREBAY</span></div>
          <div className="penstock-line" />
          <div className="powerhouse"><span>POWERHOUSE</span><i /><i /><i /></div>
          <div className="switchyard"><span>SWITCHYARD</span><i /><i /><i /><i /></div>
          <div className="tailrace"><span>TAILRACE</span></div>
          {locations.map((location, index) => (
            <button
              key={location.id}
              className={`map-marker ${location.status} ${selected?.id === location.id ? "selected" : ""}`}
              style={{ left: `${location.x}%`, top: `${location.y}%` }}
              onClick={() => onSelect(location.id)}
              aria-label={`${location.name}: ${location.status}`}
            >
              <b>{String(index + 1).padStart(2, "0")}</b>
              <span>{location.name}</span>
            </button>
          ))}
          <div className="map-scale"><span>0</span><i /><span>50 M</span></div>
        </div>
        <div className="location-strip">
          <div><span>SELECTED ASSET</span><strong>{selected?.name || "Loading plant…"}</strong></div>
          <div><span>ZONE</span><strong>{selected?.zone || "—"}</strong></div>
          <div><span>CONDITION</span><strong className={`condition ${selected?.status || "normal"}`}>{selected?.status || "Normal"}</strong></div>
          <div><span>FINDINGS</span><strong>{selected?.finding_count ?? 0}</strong></div>
        </div>
      </div>

      <aside className="activity-card">
        <div className="section-bar">
          <div><span className="section-index">02</span><strong>Latest evidence</strong></div>
          <button onClick={onShowFindings}>VIEW ALL ↗</button>
        </div>
        <div className="evidence-list">
          {findings.slice(0, 4).map((finding) => <EvidenceCard key={finding.id} finding={finding} />)}
          {!findings.length && <div className="empty-state"><b>No findings yet</b><span>Run an inspection to populate evidence.</span></div>}
        </div>
      </aside>
    </section>
  );
}

function EvidenceCard({ finding }: { finding: Finding }) {
  const [x1, y1, x2, y2] = finding.bbox;
  const box = { left: `${x1 / finding.width * 100}%`, top: `${y1 / finding.height * 100}%`, width: `${(x2 - x1) / finding.width * 100}%`, height: `${(y2 - y1) / finding.height * 100}%` };
  return (
    <article className="evidence-card">
      <div className="evidence-image">
        <img src={mediaUrl(finding.thumbnail_url)} alt={`${finding.defect_type} evidence at ${finding.location_name}`} loading="lazy" />
        <i className={`bbox ${finding.severity}`} style={box} />
        <span className={`severity-tag ${finding.severity}`}>{finding.severity}</span>
      </div>
      <div className="evidence-copy">
        <div><strong>{titleCase(finding.defect_type)}</strong><span>{Math.round(finding.confidence * 100)}% confidence</span></div>
        <p>{finding.location_name} <i /> {relativeDate(finding.created_at)}</p>
        <small className="evidence-area">{finding.affected_area_pct}% affected area</small>
      </div>
    </article>
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

function PerformanceView({ readings, pollSeconds }: { readings: PerformanceReading[]; pollSeconds: number }) {
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
            <thead><tr><th>Timestamp</th><th>Actual</th><th>Theoretical</th><th>Gap</th><th>Headwater</th><th>Tailwater</th><th>Gate position</th></tr></thead>
            <tbody>
              {newestFirst.map((reading) => (
                <tr key={reading.reading_id}>
                  <td><strong>{new Date(reading.ts).toLocaleString()}</strong><small>{relativeDate(reading.ts)}</small></td>
                  <td><b>{formatReading(reading.actual_mw, 3)}</b><small>MW</small></td>
                  <td><b>{formatReading(reading.theoretical_mw, 3)}</b><small>MW</small></td>
                  <td><b>{formatReading(reading.gap_pct, 2)}</b><small>%</small></td>
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
