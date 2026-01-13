from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import balanced_accuracy_score
import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
from collections import Counter
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

class EEGTrialsDataset(Dataset):
    def __init__(self, X, y):
        """
        X: np.ndarray, shape (N, C, T)
        y: np.ndarray or torch.Tensor, shape (N,)
        """
        self.X = X
        self.y = torch.as_tensor(y, dtype=torch.long)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx]          # (C, T), numpy
        x = torch.from_numpy(x).float()
        y = self.y[idx]
        return x, y

class EEGCroppedDataset(Dataset):
    def __init__(self, X, y, crop_len=500, n_crops_per_trial=10):
        """
        X: np.ndarray, shape (N, C, T_full)
        y: np.ndarray or torch.Tensor, shape (N,)
        """
        self.X = X
        self.y = torch.as_tensor(y, dtype=torch.long)
        self.crop_len = crop_len
        self.n_crops_per_trial = n_crops_per_trial

        self.N, self.C, self.T = X.shape
        if crop_len > self.T:
            raise ValueError(f"crop_len={crop_len} > trial length T={self.T}")

    def __len__(self):
        # each trial appears n_crops_per_trial times
        return self.N * self.n_crops_per_trial

    def __getitem__(self, idx):
        trial_idx = idx // self.n_crops_per_trial
        x_trial = self.X[trial_idx]  # (C, T)

        max_start = self.T - self.crop_len
        start = np.random.randint(0, max_start + 1)
        end = start + self.crop_len

        crop = x_trial[:, start:end]   # (C, crop_len)
        crop = torch.from_numpy(crop).float()
        label = self.y[trial_idx]
        return crop, label


class EEGFixedCenterCropDataset(Dataset):
    def __init__(self, X, y, crop_len=500):
        self.X = X
        self.y = torch.as_tensor(y, dtype=torch.long)
        self.crop_len = crop_len

        self.N, self.C, self.T = X.shape
        if crop_len > self.T:
            raise ValueError(f"crop_len={crop_len} > T={self.T}")

        # center crop
        self.start = (self.T - self.crop_len) // 2  # 250 if T=1000 and crop_len=500
        self.end = self.start + self.crop_len

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        x_trial = self.X[idx]  # (C, T)
        crop = x_trial[:, self.start:self.end]  # (C, crop_len)
        crop = torch.from_numpy(crop).float()
        label = self.y[idx]
        return crop, label



