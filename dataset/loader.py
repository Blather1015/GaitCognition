import torch
from torch.utils.data import Dataset
import numpy as np
from config import *

class GaitDataset(Dataset):
    def __init__(self, split="train"):
        X = np.load(f"{OUTPUT_DATASET}/X.npy")
        y = np.load(f"{OUTPUT_DATASET}/y.npy")

        split_idx = int(0.8 * len(X))
        if split == "train":
            self.X = X[:split_idx]
            self.y = y[:split_idx]
        else:
            self.X = X[split_idx:]
            self.y = y[split_idx:]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx]).float(),   # (75,24,3)
            torch.tensor(self.y[idx]).long()
        )
