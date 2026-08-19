# HydroVision

HydroVision is a local-first visual inspection and 2D digital twin app for
hydropower equipment. It accepts images or video, detects corrosion and leaks,
and keeps the plant map and findings register synchronized through one SQLite
snapshot.

Phase 1 operational data plumbing also records generation output, headwater,
tailwater, and gate position through a pluggable source adapter. It performs no
theoretical calculations, gap calculations, LLM calls, or VLM calls.

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

The canonical PostgreSQL DDL is in
`backend/migrations/001_performance_reading.sql`. Local development creates the
same columns in SQLite automatically; `theoretical_mw` and `gap_pct` remain
`NULL` until Phase 2.

After the service has run continuously for a day, verify the operational exit
criterion with:

```bash
python3 scripts/check_performance_soak.py --hours 24 --interval-seconds 300
```
