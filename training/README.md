# HydroVision detector training

The detector uses four object-detection datasets with bounding-box annotations:

- **Oil leakage:** `leak-kahkr-bqefk` v1, 3,427 images, one oil-leak class,
  CC BY 4.0.
- **Corrosion:** `corrosion-bi3q3` / RF100, 1,249 images, CC BY 4.0. Only
  `corrosion` boxes are mapped to HydroVision; crack/slippage-only images are
  retained as useful negative examples.
- **RGB ship corrosion:** `corrosion-detect-dataset`, 268 color images with
  YOLO boxes, MIT. The downloader verifies RGB channels, repairs four known
  reversed box dimensions, and keeps related image variants in the same split.
- **Large RGB rust/corrosion:** `rust-corrosion-detection` v13, 8,354 color
  images and four rust/corrosion severity labels, CC BY 4.0. All four labels
  are mapped to HydroVision's `corrosion` class. Polygon annotations in this
  source are converted to tight detection bounding boxes during the merge.

The compact RGB ship dataset downloads directly from Kaggle without a key.
The other sources are exported through Roboflow's API, which requires a free
private API key. The downloader checks minimum image counts, and RGB sources
are sampled and checked for red, green, and blue channels, so an incomplete or
wrong-format source cannot be mistaken for a complete dataset.

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
python3 training/prepare_yolo_dataset.py --overwrite

# Quick smoke run (proves the pipeline, not model quality):
python3 training/train_detector.py --epochs 3 --imgsz 416 --batch 8 --device cpu --name smoke

# Recommended GPU training:
python3 training/train_detector.py --epochs 80 --imgsz 640 --batch 16 --device 0
```

To add only the credential-free RGB ship dataset to sources already on disk:

```bash
python3 training/download_datasets.py \
  --skip-oil --skip-corrosion --skip-corrosion-rgb-large
python3 training/prepare_yolo_dataset.py \
  --overwrite --skip-corrosion-rgb-large
```

The first command prepares 268 RGB images in
`datasets/sources/corrosion-rgb-ship`. The second recreates the combined
train/validation/test folders in `datasets/hydrovision`; `--overwrite` does
not delete any downloaded source dataset. Rebuilds are staged and swapped only
after a complete merge, so a bad new source cannot erase the working dataset.

For Apple Silicon, replace `--device 0` with `--device mps`. For a CPU-only
machine, use `--device cpu --batch 8`; expect the 80-epoch run to take much
longer.

If an MPS training run is interrupted, resume it without resetting the
optimizer, learning-rate scheduler, or best-checkpoint tracking:

```bash
yolo detect train resume \
  model=runs/hydrovision/hydrovision-rgb-corrosion/weights/last.pt \
  device=mps
```

After training finishes, validate the selected `best.pt` on the held-out test
split using CPU and promote it to the backend only if validation succeeds:

```bash
python3 training/finalize_detector.py --device cpu --split test
```

The exported model is `models/hydrovision-yolov8n.pt`. The backend loads that
path automatically, or you can override it:

```bash
export HYDROVISION_MODEL_PATH="$PWD/models/hydrovision-yolov8n.pt"
npm run backend
```

Open `http://localhost:8001/api/health`; `detector` should be `local-yolo`.

## Production gate

Adding RGB data does not change the deployed detector until it is retrained,
and more data alone does not guarantee higher accuracy. Before operational
use, add at least 200-500 plant-camera images per class (including clean
equipment and confusing stains), annotate boxes, and keep locations/camera
sequences together when splitting to avoid data leakage. Review per-class
precision/recall and false alarms on a plant-only RGB holdout before replacing
the deployed model.
