from ..base import BaseModel
import torch.nn as nn
import timm

class ImageNetResNet(BaseModel):
    """ResNet models for ImageNet/ImageNette"""
    def __init__(
        self, 
        arch: str = "resnet18", 
        num_classes: int = 1000,
        input_size: int = 224
    ):
        super().__init__()
        self.model = timm.create_model(arch, pretrained=False)
        
        # Configure model
        self.default_cfg = {
            "num_classes": num_classes,
            "input_size": (3, input_size, input_size),
            "pool_size": (input_size//32, input_size//32),
            "crop_pct": 0.875,
            "interpolation": "bilinear",
            "mean": (0.485, 0.456, 0.406),
            "std": (0.229, 0.224, 0.225),
            "first_conv": "conv1",
            "classifier": "fc",
        }
        self.model.default_cfg = self.default_cfg
        self.model.pretrained_cfg = self.default_cfg
        
        # Modify classifier for num_classes
        feature_dim = 512 if arch in ["resnet18", "resnet34"] else 2048
        self.model.fc = nn.Linear(feature_dim, num_classes)
        
        # Set hub info for pretrained loading
        dataset = "imagenet" if num_classes == 1000 else "imagenette"
        self.hub_url = f"https://huggingface.co/timm/{arch}_{dataset}/resolve/main/pytorch_model.bin"
        self.hub_filename = f"{arch}_{dataset}.pth"

    def forward(self, x):
        return self.model(x) 