from ..base import BaseModel
import torch.nn as nn

class MNISTCNN(BaseModel):
    """CNN for MNIST/FashionMNIST classification"""
    def __init__(
        self, 
        in_channels: int = 1, 
        num_classes: int = 10
    ):
        super().__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
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
            nn.Linear(3136, 512),  # 64 * 7 * 7 = 3136
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