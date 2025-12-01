import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset.loader import GaitDataset
from model.stgcn import STGCN
from config import *
from tqdm import tqdm

def train():
    train_ds = GaitDataset("train")
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    num_classes = len(set(train_ds.y))
    model = STGCN(num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)


    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for X, y in tqdm(train_dl):
            X = X.to(device)
            y = y.to(device)


            optimizer.zero_grad()
            preds = model(X)
            loss = loss_fn(preds, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss/len(train_dl):.4f}")

    torch.save(model.state_dict(), "stgcn.pth")
    print("Model saved as stgcn.pth")

if __name__ == "__main__":
    train()
