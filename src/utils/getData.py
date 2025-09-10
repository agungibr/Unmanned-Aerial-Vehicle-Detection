import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path

class SoundDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = torch.tensor(self.features[idx][np.newaxis, ...], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return feature, label

def create_dataloaders(processed_data_path: Path, data_filename: str, batch_size: int):
    print(f"Loading {data_filename} and splitting into train/val/test sets")
    data = np.load(processed_data_path / data_filename, allow_pickle=True)
    X, y = data['X'], data['y']
    
    class_map = data['class_map'].item() if 'class_map' in data else None

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    train_dataset = SoundDataset(X_train, y_train)
    val_dataset = SoundDataset(X_val, y_val)
    test_dataset = SoundDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    input_shape = train_dataset[0][0].shape
    num_classes = len(np.unique(y))
    
    return train_loader, val_loader, test_loader, input_shape, num_classes, class_map