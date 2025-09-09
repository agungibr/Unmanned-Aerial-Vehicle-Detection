import os
import glob
import numpy as np
from tqdm import tqdm
from pathlib import Path
from extraction import FeatureExtractor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_AUDIO_PATH = PROJECT_ROOT / "data"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed"


def run_preprocessing():
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    extractor = FeatureExtractor()

    print("Processing detection data")
    detection_features = []
    detection_labels = []
    class_map_detection = {"Drone": 1, "Non-Drone": 0}

    for class_name, label in class_map_detection.items():
        audio_files = glob.glob(str(DATA_AUDIO_PATH / class_name / "*.wav"))
        for file_path in tqdm(audio_files, desc=f"Processing {class_name}"):
            try:
                feature = extractor.extract(file_path)
                detection_features.append(feature)
                detection_labels.append(label)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    np.savez(
        PROCESSED_DATA_PATH / "detection_data.npz",
        X=np.array(detection_features),
        y=np.array(detection_labels)
    )
    print("Detection data successfully saved!")

    print("\nProcessing distance estimation data")
    distance_features = []
    distance_labels = []
    distance_base_path = DATA_AUDIO_PATH / "Jarak"

    try:
        class_names_distance = sorted(
            [d.name for d in distance_base_path.iterdir() if d.is_dir()],
            key=lambda x: int(x.replace('m', ''))
        )
    except FileNotFoundError:
        print(f"Error: Folder '{distance_base_path}' not found. Skipping distance data processing.")
        class_names_distance = []

    if class_names_distance:
        class_map_distance = {name: i for i, name in enumerate(class_names_distance)}
        print(f"Discovered distance class mapping: {class_map_distance}")

        for class_name, label in class_map_distance.items():
            class_path = distance_base_path / class_name
            audio_files = glob.glob(str(class_path / "*.wav"))

            for file_path in tqdm(audio_files, desc=f"Processing {class_name}"):
                try:
                    feature = extractor.extract(file_path)
                    distance_features.append(feature)
                    distance_labels.append(label)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

        if distance_features:
            np.savez(
                PROCESSED_DATA_PATH / "distance_data.npz",
                X=np.array(distance_features),
                y=np.array(distance_labels),
                class_map=class_map_distance
            )
            print(f"Distance data successfully saved to '{PROCESSED_DATA_PATH / 'distance_data.npz'}'")
        else:
            print("No distance features were extracted.")

if __name__ == '__main__':
    run_preprocessing()