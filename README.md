## UAV Sound Detection — Usage Guide

This project provides scripts to preprocess audio, train CNN models, evaluate them, and run single-file predictions for drone vs non-drone and distance classes.

## Requirements

- Python 3.12
- Linux bash shell

Install dependencies:

```bash
pip install -r requirements.txt
```

## Data layout (matches current code)

The preprocessing script expects WAV files under the `src/data/` folder:

- `src/data/Drone/*.wav`
- `src/data/Non-Drone/*.wav`
- `src/data/Jarak/<class>/*.wav` (e.g., `src/data/Jarak/m5/*.wav`, `m10`, ... for distance classes)

Tip: If your data currently lives under `data/raw/`, either move/copy it into `src/data/` or create symlinks.

Also, the training/eval scripts read `config.yaml` relative to the `src/` folder. Ensure a copy is available at:

- `src/config.yaml`

Example copy command (run from project root):

```bash
cp config.yaml src/config.yaml
```

## 1) Preprocess features

Converts WAVs into log-mel spectrogram arrays and stores NPZ files used by training/testing.

Run from the project root:

```bash
python -m src.data.preprocessing
```

Outputs (under `src/data/processed/`):
- `detection_data.npz` (X, y)
- `distance_data.npz` (X, y, class_map) if `src/data/Jarak/` exists

## 2) Train

Train either the detection model or the distance model. Run from the project root:

```bash
python -m src.models.train --model_type detection
python -m src.models.train --model_type distance
```

Artifacts (saved relative to `src/`):
- Models: `src/models/*.pth`
- Training curves: `src/results/<model>_history_plot.png`

## 3) Test / Evaluate

Evaluate on the test split and save a confusion matrix plot. Run from the project root:

```bash
python -m src.models.test --model_type detection
python -m src.models.test --model_type distance
```

Artifacts:
- Confusion matrix: `src/results/<model>_confusion_matrix.png`

## 4) Predict a single WAV

The current CLI in `src/models/predict.py` is positional-only and may error if run directly. Use this Python one-liner instead (replace `path/to/audio.wav`):

```bash
python - << 'PY'
from pathlib import Path
import yaml
from src.models.predict import predict_sound

config = yaml.safe_load(open('config.yaml', 'r'))
predict_sound(Path('path/to/audio.wav'), config)
PY
```

Behavior:
- If detection predicts Drone, it will also estimate the distance class using the distance model.

## Notes

- Default hyperparameters and file names live in `config.yaml`. The scripts read the copy at `src/config.yaml`.
- Feature extraction defaults to 5 seconds padded/truncated per clip (see `src/features/extraction.py`).
- For GPU training, ensure a CUDA-enabled PyTorch install.

