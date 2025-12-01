import torch
import torch.nn as nn

class STGCNBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.spatial = nn.Conv2d(in_c, out_c, kernel_size=1)
        self.temporal = nn.Conv2d(out_c, out_c, kernel_size=(9,1), padding=(4,0))
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.spatial(x))
        x = self.relu(self.temporal(x))
        return x

class STGCN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.block1 = STGCNBlock(3, 64)
        self.block2 = STGCNBlock(64, 128)
        self.block3 = STGCNBlock(128, 256)
        self.fc = nn.Linear(256 * 24, num_classes)

    def forward(self, x):
        # x shape: (N, T, V, C)
        x = x.permute(0,3,1,2)  # (N,C,T,V)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = x.mean(dim=2)  # temporal pooling → (N,256,V)
        x = x.reshape(x.size(0), -1)
        return self.fc(x)
