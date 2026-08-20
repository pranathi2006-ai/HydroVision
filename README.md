# HydroVision

HydroVision is a local-first visual inspection and 2D digital twin app for
hydropower equipment. It accepts images or video, detects corrosion and leaks,
and keeps the plant map and findings register synchronized through one SQLite
snapshot.

The performance pipeline records generation output, headwater, tailwater, and
gate position through a pluggable source adapter. Phase 2 uses static imported
OEM/design curves to populate healthy-condition theoretical output and the raw
performance gap. It makes no LLM or VLM calls.

## What is enforced

- Inference runs locally. Set `HYDROVISION_MODEL_PATH` to fine-tuned YOLO
  weights; a deterministic OpenCV baseline is used until weights are supplied.
- Video is sampled every 2.5 seconds. A 30-second clip produces 12 sampled
  frames, never hundreds of full-rate calls.
- Every source image and sampled frame is SHA-256 hashed before inference.
  Duplicate content reuses the SQLite cache.
- Images are orientation-corrected, resized to a maximum 1024px long edge, and
  stored as compressed JPEG evidence.
- Uncached images are inferred as a batch. No LLM, VLM, cloud store, or paid
  per-image API is called.
- Thumbnails use native browser lazy loading.

## Run locally

Prerequisites: Node.js 22+, Python 3.9+, and the Python packages in
`backend/requirements.txt`.

```bash
npm install
python3 -m pip install -r backend/requirements.txt
./scripts/dev.sh
```

The web app runs at `http://localhost:3000`; the local API and documentation
run at `http://localhost:8001` and `http://localhost:8001/docs`.

Both services listen on the local network. To open HydroVision from another
computer on the same Wi-Fi, find this machine's private IPv4 address and open
`http://<private-ip>:3000`. Keep the terminal running, and allow incoming
connections if the operating-system firewall asks.

To run the two processes separately:

```bash
npm run dev
npm run backend
```

## Model weights

The repository includes reproducible acquisition, label normalization,
training, and evaluation commands for a two-class YOLOv8n detector. See
[`training/README.md`](training/README.md). The backend automatically loads
`models/hydrovision-yolov8n.pt` after training, or you can point it at another
compatible model:

```bash
export HYDROVISION_MODEL_PATH=/absolute/path/to/best.pt
npm run backend
```

Inference is local and forced to CPU with a maximum image size of 1024px. Model
training downloads the selected public data and pretrained weights only when
you explicitly run the training commands.

## Verification

```bash
python3 -m pytest backend/tests/test_pipeline.py -q
npm test
```

The Python tests cover resize enforcement, 30-second video sampling, and a
known corrosion patch with an expected bounding box and confidence. `npm test`
builds the web app and checks the server-rendered product shell.

Local evidence and cache rows live under `data/`. CSV export is
available from the app header or at `/api/export.csv`.

## Operational reading source

The default `MockSourceAdapter` emits smooth, plausible readings every five
minutes. The verification page's **Readings** section fetches the same raw rows
on a minutes-scale interval. To switch the entire ingestion path to the plant
source, change one setting:

```bash
export HYDROVISION_PERFORMANCE_SOURCE=real
```

Before doing that, copy the `HYDROVISION_PLANT_*` connection stub from
`.env.example` and confirm the endpoint, credentials, JSON field names, source
update interval, units, and physical validation limits with the controls team.
`RealSourceAdapter` currently implements a JSON/HTTP transport; if the confirmed
interface is OPC-UA or a historian SDK, replace only that adapter's transport
body and retain its `getLatestReading()` return contract.

The canonical PostgreSQL DDL is in `backend/migrations/`. Local development
creates the same columns and reference-curve tables in SQLite automatically.

## Phase 2 reference curves

Reference values are not hardcoded in the calculation service. Import a JSON
object or a directory containing these three editable CSV files:

- `turbine_performance_curve.csv`
- `gate_flow_curve.csv`
- `hydraulic_loss_baseline.csv`

The headers match the columns in `backend/migrations/002_reference_curves.sql`.
Importing replaces the prior dataset atomically, validates complete grids and
physical ranges, caches new SciPy interpolators, and recalculates every stored
performance row:

```bash
python3 scripts/import_reference_curves.py /path/to/oem-curves \
  --dataset-name "OEM hill curves revision C" \
  --unit-id turbine_1 \
  --nameplate-mw 75
```

The bundled `reference_curves/mock_design/` files exist only to exercise the
mock adapter and verification chart. They are marked as demo data in the
database and are refused when `HYDROVISION_PERFORMANCE_SOURCE=real`. Import the
plant's OEM/design documents before activating the real source, then restart
the backend so its in-memory interpolation cache uses the new dataset.

New readings are calculated before insert. If a curve lookup is out of range or
invalid, the reading is rejected and logged rather than stored with missing
Phase 2 fields. Output above the configured nameplate capacity is logged and
clamped before the gap is calculated.

After the service has run continuously for a day, verify the operational exit
criterion with:

```bash
python3 scripts/check_performance_soak.py --hours 24 --interval-seconds 300 --nameplate-mw 75
```

## Phase 3 site detectors

The existing upload endpoint now routes each location to an exact asset/sensor
mapping and writes a `detection_event` for every applicable detector result,
including explicit healthy results. The routing is:

- intake gate: trash-rack blockage area plus geometric gate-position mismatch
- penstock valve: the existing local oil-leak and corrosion heads
- each turbine: cavitation/pitting wear area
- draft tube: the existing corrosion head plus blockage area
- main transformer: thermal ΔT with three-frame persistence

All paths are local OpenCV or optional local YOLO weights. Uploads retain the
existing SHA-256 cache and 2.5-second video sampling. `GET /api/detection-events`
returns the structured events, while `measurement` contains values such as
`blockage_pct`, visual/commanded gate position, pitting area, or thermal ΔT.

The public-data provenance registry is
`training/phase3_dataset_registry.json`. Register actual downloaded images and
their hashes without copying pixels into the database:

```bash
python3 scripts/import_training_dataset.py taco /path/to/TACO --split train
```

Use `training/generate_phase3_synthetic.py` to generate deterministic,
mask-preserving debris or pitting composites from locally licensed source
images. Synthetic output is never generated or parsed during inference.

## Phase 4 rule attribution

Readings above `HYDROVISION_ATTRIBUTION_GAP_THRESHOLD_PCT` are matched only to
the closest prior active Phase 3 evidence inside the configured time window.
Geometric rack/gate rules and lower-confidence severity maps are stored in
`attribution_rule_config`, so plant experts can tune coefficients without code
changes. Results retain a required `event_id`, use `method=rule_based`, and are
ranked by estimated MW while displaying confidence separately. Triggering gaps
with no evidence are recorded and logged as unexplained.

`actual_mw` must be measured at the generator terminal. Mock data declares that
boundary explicitly; real-source attribution remains disabled until this is
confirmed:

```bash
export HYDROVISION_ACTUAL_MW_METER_LOCATION=generator_terminal
```

Use `grid_connection` if that is the real boundary; transformer loss then needs
a separately scoped design. Transformer thermal events are never candidates in
the Phase 4 hydraulic ranking. Query a ranked result with
`GET /api/performance/attribution?reading_id=<id>`.
