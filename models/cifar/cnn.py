from ..base import BaseModel
import torch.nn as nn

class CIFAR10CNN(BaseModel):
    """CNN for CIFAR10 classification"""
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=False),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.05),
        )

        self.fc_layer = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(4096, 512),
            nn.ReLU(inplace=False),
            nn.Linear(512, 256),
            nn.ReLU(inplace=False),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        out = self.conv_layer(x)
        out = out.view(out.size(0), -1)
        out = self.fc_layer(out)
        return out 