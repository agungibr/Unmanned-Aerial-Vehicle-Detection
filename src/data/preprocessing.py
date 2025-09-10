import yaml
import glob
import h5py
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

    experiments = ['detection', 'distance'] 
    for experiment_name in experiments:
        params = config[experiment_name]
        save_path = PROCESSED_DATA_PATH / params['data_filename']

        if experiment_name == 'detection':
            class_map = {"Drone": 1, "Non-Drone": 0}
            base_path = RAW_DATA_PATH
        else:
            base_path = RAW_DATA_PATH / "Jarak"
            class_map = {d.name: i for i, d in enumerate(sorted(
                [d for d in base_path.iterdir() if d.is_dir()],
                key=lambda x: int(x.name.replace('m', ''))
            ))}

        feature_list, label_list = [], []
        for class_name, label in class_map.items():
            class_path = base_path / class_name
            audio_files = glob.glob(str(class_path / "*.wav"))
            for file_path in tqdm(audio_files, desc=f"Processing {class_name}"):
                try:
                    feature = extractor.extract(file_path, method=params['feature_method'])
                    feature_list.append(feature)
                    label_list.append(label)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
        
        with h5py.File(save_path, 'w') as hf:
            hf.create_dataset('X', data=np.array(feature_list))
            hf.create_dataset('y', data=np.array(label_list))
            if 'distance' in experiment_name:
                hf.attrs['class_map'] = str(class_map)
        
        print(f"Successfully saved data to {save_path}")