import ast
import yaml
import torch
import argparse
import numpy as np
from pathlib import Path
from .CNN import CNN
from .LSTM import LSTMModel
from ..features.extraction import FeatureExtractor

def predict_sound(audio_file_path: Path, config: dict):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} for prediction.")

    detection_params = config['detection']
    distance_params = config['distance']
    default_params = config['default']
    
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    MODELS_PATH = PROJECT_ROOT / default_params['model_save_path']
    PROCESSED_DATA_PATH = PROJECT_ROOT / default_params['data_path']

    extractor = FeatureExtractor()
    detection_feature = extractor.extract(audio_file_path, method=detection_params['feature_method'])
    det_input_shape = (1, detection_feature.shape[0], detection_feature.shape[1])
    detection_model = CNN(det_input_shape, num_classes=1).to(device)
    detection_model.load_state_dict(torch.load(MODELS_PATH / detection_params['model_filename'], map_location=device))
    detection_model.eval()

    det_feature_tensor = torch.tensor(detection_feature[np.newaxis, ...], dtype=torch.float32).to(device)
    with torch.no_grad():
        detection_output = detection_model(det_feature_tensor)
        is_drone = torch.sigmoid(detection_output).item() > 0.5

    if is_drone:
        print("Drone Detected. Loading Distance Model (LSTM)")
        distance_feature = extractor.extract(audio_file_path, method=distance_params['feature_method'])
        dist_input_shape = (distance_feature.shape[0], distance_feature.shape[1])
        
        with h5py.File(PROCESSED_DATA_PATH / distance_params['data_filename'], 'r') as hf:
            class_map_str = hf.attrs.get('class_map', '{}')
            class_map = ast.literal_eval(class_map_str) if class_map_str else None
        idx_to_class = {v: k for k, v in class_map.items()}
        num_distance_classes = len(idx_to_class)

        distance_model = LSTMModel(
            input_size=dist_input_shape[1], 
            hidden_size=distance_params['lstm_hidden_size'], 
            num_layers=distance_params['lstm_num_layers'], 
            num_classes=num_distance_classes
        ).to(device)
        distance_model.load_state_dict(torch.load(MODELS_PATH / distance_params['model_filename'], map_location=device))
        distance_model.eval()
        
        dist_feature_tensor = torch.tensor(distance_feature, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            distance_output = distance_model(dist_feature_tensor)
            predicted_idx = torch.argmax(distance_output, dim=1).item()
            predicted_distance = idx_to_class[predicted_idx]

        print("\nPrediction Result")
        print(f"Status: Drone Detected")
        print(f"Estimated Distance: {predicted_distance}")

    else:
        print("\nPrediction Result")
        print(f"Status: Non-Drone Detected")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict if a sound is a drone and estimate its distance.")
    parser.add_argument('--input_file', type=str, required=True, help="Path to the input .wav audio file.")
    args = parser.parse_args()

    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file not found at '{input_path}'")
    else:
        config_path = Path(__file__).resolve().parents[2] / "config.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        predict_sound(input_path, config)