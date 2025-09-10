import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from pathlib import Path

class SoundDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = torch.tensor(self.features[idx][np.newaxis, ...], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32).unsqueeze(0)
        return feature, label

def create_dataloaders(data_path: Path, batch_size: int):
    print("Loading preprocessed data and splitting into train/val/test sets")
    data = np.load(data_path)
    X, y = data['X'], data['y']

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
    
    return train_loader, val_loader, test_loader, input_shape