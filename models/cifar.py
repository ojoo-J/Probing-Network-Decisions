import torch
import torch.nn as nn
from .base import BaseModel

class CIFAR10CNN(BaseModel):
    """Simple CNN for CIFAR10"""
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class CIFARResNet(BaseModel):
    """ResNet for CIFAR"""
    def __init__(self, arch="resnet18", num_classes=10):
        super().__init__()
        import timm
        self.model = timm.create_model(
            arch, 
            pretrained=False,
            num_classes=num_classes,
            in_chans=3
        )
        
    def forward(self, x):
        return self.model(x) 