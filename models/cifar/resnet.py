from ..base import BaseModel
import torch.nn as nn
import timm

class CIFARResNet(BaseModel):
    """ResNet models for CIFAR"""
    def __init__(self, arch: str = "resnet18", num_classes: int = 10):
        super().__init__()
        self.model = timm.create_model(arch, pretrained=False)
        
        # Configure for CIFAR
        self.default_cfg = {
            "num_classes": num_classes,
            "input_size": (3, 32, 32),
            "pool_size": (4, 4),
            "crop_pct": 1,
            "interpolation": "bilinear",
            "first_conv": "conv1",
            "classifier": "fc",
        }
        self.model.default_cfg = self.default_cfg
        self.model.pretrained_cfg = self.default_cfg
        
        # Modify first conv and remove maxpool for CIFAR
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()
        
        # Modify classifier
        feature_dim = 512 if arch in ["resnet18", "resnet34"] else 2048
        self.model.fc = nn.Linear(feature_dim, num_classes)
        
        # Set hub info for pretrained loading
        dataset = "cifar10" if num_classes == 10 else "cifar100"
        self.hub_url = f"https://huggingface.co/edadaltocg/{arch}_{dataset}/resolve/main/pytorch_model.bin"
        self.hub_filename = f"{arch}_{dataset}.pth"

    def forward(self, x):
        return self.model(x) 