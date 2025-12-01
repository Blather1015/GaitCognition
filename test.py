import torch
from torch.utils.data import DataLoader
from dataset.loader import GaitDataset
from model.stgcn import STGCN
from config import *

def test():
    test_ds = GaitDataset("test")
    test_dl = DataLoader(test_ds, batch_size=1)

    # Auto-select device (CPU for your laptop)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    num_classes = 8


    # Create model on the detected device
    model = STGCN(num_classes).to(device)

    # Load weights
    model.load_state_dict(torch.load("stgcn.pth", map_location=device))
    model.eval()

    correct = 0

    with torch.no_grad():
        for X, y in test_dl:
            X = X.to(device)
            y = y.to(device)

            pred = model(X).argmax(dim=1)
            correct += (pred == y).sum().item()

    print("Accuracy:", correct / len(test_ds))


if __name__ == "__main__":
    test()
