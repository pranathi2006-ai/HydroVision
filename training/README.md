# HydroVision detector training

The detector uses two object-detection datasets with bounding-box annotations:

- **Oil leakage:** `leak-kahkr-bqefk` v1, 3,427 images, one oil-leak class,
  CC BY 4.0.
- **Corrosion:** `corrosion-bi3q3` / RF100, 1,249 images, CC BY 4.0. Only
  `corrosion` boxes are mapped to HydroVision; crack/slippage-only images are
  retained as useful negative examples.

Both public datasets are exported through Roboflow's official API, which
requires a free private API key. The downloader checks image counts so a
partial download cannot be mistaken for a complete dataset.

Do not use the Mendeley wind-turbine oil dataset as the primary YOLO source:
it is a useful synthetic classification dataset, but it has no ground-truth
bounding boxes.

## Exact commands

Run from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r training/requirements.txt

# Create a free Roboflow account, copy the private API key from Settings > API Keys,
# and keep it only in this shell (never put it in a committed file).
export ROBOFLOW_API_KEY='replace-with-your-private-key'

python3 training/download_datasets.py
python3 training/prepare_yolo_dataset.py

# Quick smoke run (proves the pipeline, not model quality):
python3 training/train_detector.py --epochs 3 --imgsz 416 --batch 8 --device cpu --name smoke

# Recommended GPU training:
python3 training/train_detector.py --epochs 80 --imgsz 640 --batch 16 --device 0
```

For Apple Silicon, replace `--device 0` with `--device mps`. For a CPU-only
machine, use `--device cpu --batch 8`; expect the 80-epoch run to take much
longer.

The exported model is `models/hydrovision-yolov8n.pt`. The backend loads that
path automatically, or you can override it:

```bash
export HYDROVISION_MODEL_PATH="$PWD/models/hydrovision-yolov8n.pt"
npm run backend
```

Open `http://localhost:8000/api/health`; `detector` should be `local-yolo`.

## Production gate

These public datasets establish the pipeline, not production accuracy on your
plant. Before operational use, add at least 200-500 plant-camera images per
class (including clean equipment and confusing stains), annotate boxes, and
keep locations/camera sequences together when splitting to avoid data leakage.
Review per-class precision/recall and false alarms on a plant-only holdout.
