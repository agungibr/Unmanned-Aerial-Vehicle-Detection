import ast
import h5py
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

class H5Dataset(Dataset):
    def __init__(self, h5_path, indices, model_architecture):
        self.h5_path = h5_path
        self.indices = indices
        self.model_architecture = model_architecture
        
        self.h5_file = h5py.File(self.h5_path, 'r')
        self.features = self.h5_file['X']
        self.labels = self.h5_file['y']

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        item_idx = self.indices[idx]
        
        feature = self.features[item_idx]
        label = self.labels[item_idx]
        
        if self.model_architecture == 'cnn':
            feature_tensor = torch.tensor(feature[np.newaxis, ...], dtype=torch.float32)
        else:
            feature_tensor = torch.tensor(feature, dtype=torch.float32)
        
        label_tensor = torch.tensor(label, dtype=torch.long)
        return feature_tensor, label_tensor

def create_dataloaders(processed_data_path: Path, data_filename: str, batch_size: int, model_architecture: str):
    h5_path = processed_data_path / data_filename
    
    with h5py.File(h5_path, 'r') as hf:
        total_samples = len(hf['y'])
        labels = hf['y'][:]
        class_map_str = hf.attrs.get('class_map', '{}')
        class_map = ast.literal_eval(class_map_str) if class_map_str else None

    indices = np.arange(total_samples)
    
    train_indices, temp_indices, _, _ = train_test_split(
        indices, labels, test_size=0.2, random_state=42, stratify=labels
    )
    val_indices, test_indices, _, _ = train_test_split(
        temp_indices, labels[temp_indices], test_size=0.5, random_state=42, stratify=labels[temp_indices]
    )

    train_dataset = H5Dataset(h5_path, train_indices, model_architecture)
    val_dataset = H5Dataset(h5_path, val_indices, model_architecture)
    test_dataset = H5Dataset(h5_path, test_indices, model_architecture)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    input_shape = train_dataset[0][0].shape
    num_classes = len(np.unique(labels))
    
    return train_loader, val_loader, test_loader, input_shape, num_classes, class_map