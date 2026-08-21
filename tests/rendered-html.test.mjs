import assert from "node:assert/strict";
import test from "node:test";
import { readFile } from "node:fs/promises";

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);
  return worker.fetch(
    new Request("http://localhost/", { headers: { accept: "text/html" } }),
    { ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) } },
    { waitUntil() {}, passThroughOnException() {} },
  );
}

test("server-renders the unified Phase 5 operations dashboard", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  const html = await response.text();
  assert.match(html, /<title>HydroVision/);
  assert.match(html, /Condition and energy impact/);
  assert.match(html, /map and waterfall will load together/i);
  assert.match(html, /New inspection/);
  assert.doesNotMatch(html, /codex-preview|react-loading-skeleton|Your site is taking shape/i);
});

test("map and waterfall share one current-dashboard fetch and selection state", async () => {
  const source = await readFile(new URL("../app/HydroVisionApp.tsx", import.meta.url), "utf8");
  assert.equal(source.match(/fetch\(`\$\{API\}\/api\/dashboard\/current`\)/g)?.length, 1);
  const spatial = source.slice(source.indexOf("function SpatialTwin"), source.indexOf("function EnergyWaterfall"));
  const waterfall = source.slice(source.indexOf("function EnergyWaterfall"), source.indexOf("function SiteDetailPanel"));
  assert.doesNotMatch(spatial, /fetch\(/);
  assert.doesNotMatch(waterfall, /fetch\(/);
  assert.match(source, /selectedAssetId=\{selectedAssetId\}/);
  assert.match(source, /onSelect=\{setSelectedAssetId\}/);
  assert.match(source, /Automated attribution verification monitoring/);
  assert.match(source, /inconclusive_null_pct/);
});

test("detail panel identifies the stored attribution method without model calls", async () => {
  const source = await readFile(new URL("../app/HydroVisionApp.tsx", import.meta.url), "utf8");
  assert.match(source, /<dt>Estimate method<\/dt>/);
  assert.match(source, /titleCase\(attribution\.method\)/);
  const detail = source.slice(source.indexOf("function SiteDetailPanel"), source.indexOf("function FindingsView"));
  assert.doesNotMatch(detail, /fetch\(|useEffect\(/);
});
