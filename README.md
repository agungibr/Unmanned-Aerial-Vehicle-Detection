## UAV Sound Detection and Positioning

This project provides a complete pipeline to train and use a two-stage machine learning system for audio-based drone surveillance.

1. Detection Model: A CNN that classifies sounds as `Drone` or `Non-Drone`.
2. Position Model: A secondary model (either CNN or MLP) that estimates the distance of a sound, used only after it has been classified as a drone.

The entire workflow is managed by a central configuration file (`config.yaml`) and can be run from the command line.

---
## Project Structure

Unmanned-Aerial-Vehicle-Detection/
├── data/
│   ├── raw/         # <-- PLACE YOUR .WAV FILES HERE
│   └── processed/   # Automatically generated
├── models/          # Saved .pth models are stored here
├── results/         # Output plots and metrics are saved here
├── src/             # All source code
│   ├── data/
│   ├── features/
│   ├── models/
│   └── utils/
├── config.yaml      # <-- CONFIGURE YOUR EXPERIMENTS HERE
└── requirements.txt


---
## Setup

First, set up the project environment. Using a virtual environment is highly recommended.

1. Clone the Repository
	```bash
	git clone <your-repo-url>
	cd Unmanned-Aerial-Vehicle-Detection
	```

2. Create and Activate a Virtual Environment

	- On macOS/Linux:
		```bash
		python3 -m venv env
		source env/bin/activate
		```
	- On Windows:
		```bash
		python -m venv env
		.\env\Scripts\activate
		```

3. Install Dependencies
	```bash
	pip install -r requirements.txt
	```

---
## Workflow / Usage

All commands should be run from the project root directory.

### Place the Data

Place your raw audio files in the `data/raw/` directory using the following structure:
- `data/raw/Drone/*.wav`
- `data/raw/Non-Drone/*.wav`
- `data/raw/Jarak/<distance>/*.wav` (e.g., `data/raw/Jarak/1m/`, `data/raw/Jarak/2m/`, etc.)

### Preprocess Data

This script reads the raw audio, extracts features as defined in `config.yaml`, and saves the processed datasets to `data/processed/`.

```bash
python -m src.data.preprocessing
```

### Train Models

Train any of the models defined in your `config.yaml` using the `--experiment` flag.

- Train the Detection Model (CNN):

```bash
python -m src.models.train --experiment detection
```

- Train the Distance Model (CNN):

```bash
python -m src.models.train --experiment distance_cnn
```

- Train the Distance Model (MLP):

```bash
python -m src.models.train --experiment distance_mlp
```

### Step 4: Evaluate Models

Evaluate a trained model on the test set to generate a confusion matrix and accuracy score.

- Evaluate the Detection Model:

```bash
python -m src.models.test --model_type detection
```

- Evaluate the Distance Model (CNN):

```bash
python -m src.models.test --model_type distance
```

Note: Evaluation for the MLP distance model is not wired in the current `test.py` (it uses the CNN architecture). You can still train the MLP; for evaluation, either adapt `test.py` or export logits and compute metrics manually.

### Step 5: Make a Prediction

Use the prediction utility to get a real-time prediction on a single audio file. The script automatically uses the two-step pipeline.

Current CLI note: the `predict.py` positional-argument parser needs a small tweak before direct CLI use. Until then, use this snippet (replace the path):

```bash
python - << 'PY'
from pathlib import Path
import yaml
from src.models.predict import predict_sound

with open('config.yaml', 'r') as f:
	config = yaml.safe_load(f)

predict_sound(Path('data/raw/Jarak/6m/some_drone.wav'), config)
PY
```

Example 1 (A Drone is Detected):

Expected Output (example):

```
Using cpu for prediction.

✅ --- Prediction Result --- ✅
Status: Drone Detected
Estimated Distance: 6m
```

Example 2 (No Drone is Detected):

```
Using cpu for prediction.

❌ --- Prediction Result --- ❌
Status: Non-Drone Detected
```

---
## Configuration

All project settings, hyperparameters, and file paths can be modified in the central `config.yaml` file without needing to change the source code. This is where you can adjust epochs, learning rates, batch sizes, and define new experiments (`detection`, `distance_cnn`, `distance_mlp`).

Models and results are saved to the folders defined under the `default` section:
- `default.data_path`: processed datasets location (default: `data/processed/`)
- `default.model_save_path`: model checkpoint directory (default: `models/`)
- `default.results_save_path`: plots and metrics directory (default: `results/`)

---
## Notes

- Feature extraction methods are configured per experiment (see `feature_method` in `config.yaml`).
- GPU will be used automatically if available (PyTorch CUDA build required).
- Ensure your WAVs are readable and of sufficient length; the extractor handles trimming/padding internally.

