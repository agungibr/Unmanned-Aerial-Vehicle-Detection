import torch
import yaml
import argparse
import numpy as np
from pathlib import Path
from models.CNN import CNN
from src.features.extraction import FeatureExtractor

def predict_sound(audio_file_path: Path, config: dict):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} for prediction.")

    detection_params = config['detection']
    distance_params = config['distance']
    default_params = config['default']
    
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    MODELS_PATH = PROJECT_ROOT / default_params['model_save_path']
    PROCESSED_DATA_PATH = PROJECT_ROOT / default_params['data_path']

    distance_data = np.load(PROCESSED_DATA_PATH / distance_params['data_filename'], allow_pickle=True)
    class_map_distance = distance_data['class_map'].item()
    idx_to_class = {v: k for k, v in class_map_distance.items()}
    num_distance_classes = len(idx_to_class)

    extractor = FeatureExtractor()
    feature = extractor.extract(audio_file_path)
    input_shape = (1, feature.shape[0], feature.shape[1])
    feature_tensor = torch.tensor(feature[np.newaxis, ...], dtype=torch.float32).to(device)

    detection_model = CNN(input_shape, num_classes=1).to(device)
    distance_model = CNN(input_shape, num_classes=num_distance_classes).to(device)

    detection_model.load_state_dict(torch.load(MODELS_PATH / detection_params['model_filename'], map_location=device))
    distance_model.load_state_dict(torch.load(MODELS_PATH / distance_params['model_filename'], map_location=device))

    detection_model.eval()
    distance_model.eval()

    with torch.no_grad():
        detection_output = detection_model(feature_tensor)
        is_drone = torch.sigmoid(detection_output).item() > 0.5

        if is_drone:
            distance_output = distance_model(feature_tensor)
            distance_probs = torch.softmax(distance_output, dim=1)
            predicted_idx = torch.argmax(distance_probs, dim=1).item()
            predicted_distance = idx_to_class[predicted_idx]

            print("\nPrediction Result")
            print(f"Status: Drone Detected")
            print(f"Estimated Distance: {predicted_distance}")
        else:
            print("\nPrediction Result")
            print(f"Status: Non-Drone Detected")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict if a sound is a drone and estimate its distance.")
    parser.add_argument('input_file', type=str, required=True,
                        help="Path to the input .wav audio file.")
    args = parser.parse_args()

    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file not found at '{input_path}'")
    else:
        config_path = Path(__file__).resolve().parents[2] / "config.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        predict_sound(input_path, config)