import os
import glob
import yaml
import numpy as np
from tqdm import tqdm
from pathlib import Path
from ..features.extraction import FeatureExtractor

if __name__ == '__main__':
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    with open(PROJECT_ROOT / "config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw"
    PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed"
    PROCESSED_DATA_PATH.mkdir(exist_ok=True)
    
    extractor = FeatureExtractor()
    experiments = ['detection', 'distance_cnn', 'distance_mlp']

    for experiment_name in experiments:
        params = config[experiment_name]
        print(f"\nProcessing data for '{experiment_name}' experiment")
        
        features, labels = [], []
        if experiment_name == 'detection':
            class_map = {"Drone": 1, "Non-Drone": 0}
            base_path = RAW_DATA_PATH
            folder_names = class_map.keys()
        else:
            base_path = RAW_DATA_PATH / "Jarak"
            class_map = {d.name: i for i, d in enumerate(sorted(
                [d for d in base_path.iterdir() if d.is_dir()],
                key=lambda x: int(x.name.replace('m', ''))
            ))}
            folder_names = class_map.keys()

        for class_name, label in class_map.items():
            class_path = base_path / class_name
            audio_files = glob.glob(str(class_path / "*.wav"))
            for file_path in tqdm(audio_files, desc=f"Processing {class_name}"):
                try:
                    feature = extractor.extract(file_path, method=params['feature_method'])
                    features.append(feature)
                    labels.append(label)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

        save_path = PROCESSED_DATA_PATH / params['data_filename']
        
        if 'distance' in experiment_name:
            np.savez(save_path, X=np.array(features, dtype=object), y=np.array(labels), class_map=class_map)
        else:
            np.savez(save_path, X=np.array(features, dtype=object), y=np.array(labels))
            
        print(f"Successfully saved data to {save_path}")