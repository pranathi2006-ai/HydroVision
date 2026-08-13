# HydroVision

HydroVision is a local-first visual inspection and 2D digital twin app for
hydropower equipment. It accepts images or video, detects corrosion and leaks,
and keeps the plant map and findings register synchronized through one SQLite
snapshot.

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
run at `http://localhost:8000` and `http://localhost:8000/docs`.

To run the two processes separately:

```bash
npm run dev
npm run backend
```

## Model weights

Export a YOLO model whose class names contain `rust`/`corrosion` and
`leak`/`oil`/`water`, then point the backend at it:

```bash
export HYDROVISION_MODEL_PATH=/absolute/path/to/best.pt
npm run backend
```

The model is loaded from disk and forced to CPU-local inference with a maximum
image size of 1024px. The repository does not silently download generic weights.

## Verification

```bash
python3 -m pytest backend/tests/test_pipeline.py -q
npm test
```

The Python tests cover resize enforcement, 30-second video sampling, and a
known corrosion patch with an expected bounding box and confidence. `npm test`
builds the web app and checks the server-rendered product shell.

Local evidence, cache rows, and review labels live under `data/`. CSV export is
available from the app header or at `/api/export.csv`.
