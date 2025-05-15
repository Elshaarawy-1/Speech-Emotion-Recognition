import os
import torch
from torch.utils.data import Dataset
import numpy as np

class AudioEmotionDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]
    
        # Load dictionary with features and label
        features = np.load(file_path, allow_pickle=True)
        if self.transform:
            features = self.transform(features)

        # Convert to torch tensors
        features = torch.tensor(features, dtype=torch.float32)
        return features, label