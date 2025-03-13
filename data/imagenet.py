from typing import Dict, Any, Tuple, Optional
import os
import torch
import torchvision
import torchvision.transforms as transforms
from .base import BaseDataModule

class ImageNetteDataModule(BaseDataModule):
    """ImageNette data module"""
    def __init__(
        self, 
        data_dir: str, 
        batch_size: int = 32,
        normalize: bool = False,
        input_size: int = 128
    ):
        super().__init__(data_dir, batch_size)
        self.normalize = normalize
        self.input_size = input_size
        
    def _get_data_info(self) -> Dict[str, Any]:
        """Get dataset information from get_data.py"""
        data_mean = [0.485, 0.456, 0.406]  # ImageNet stats
        data_std = [0.229, 0.224, 0.225]
        
        if not self.normalize:
            data_mean = [0.0, 0.0, 0.0]
            data_std = [1.0, 1.0, 1.0]
            
        return {
            "data_set": "ImageNette",
            "data_shape": [3, self.input_size, self.input_size],
            "n_bits": 8,
            "temp": 1,
            "num_classes": 10,
            "class_names": [
                "tench",
                "English springer",
                "cassette player",
                "chain saw",
                "church",
                "French horn",
                "garbage truck",
                "gas pump",
                "golf ball",
                "parachute",
            ],
            "data_mean": data_mean,
            "data_std": data_std,
        }
        
    def setup(self):
        """Set up the datasets"""
        transforms_list = [
            transforms.CenterCrop(self.input_size),
            transforms.ToTensor(),
        ]
        
        if self.normalize:
            transforms_list.append(
                transforms.Normalize(
                    mean=self.info["data_mean"],
                    std=self.info["data_std"]
                )
            )
            
        transform = transforms.Compose(transforms_list)
        
        self.train_dataset = torchvision.datasets.ImageFolder(
            root=os.path.join(self.data_dir, "train"),
            transform=transform
        )
        
        self.val_dataset = torchvision.datasets.ImageFolder(
            root=os.path.join(self.data_dir, "val"),
            transform=transform
        )


class ImageNetDataModule(ImageNetteDataModule):
    """ImageNet data module"""
    def __init__(
        self, 
        data_dir: str, 
        batch_size: int = 32,
        normalize: bool = False,
        input_size: int = 224
    ):
        super().__init__(data_dir, batch_size, normalize, input_size)
        
    def _get_data_info(self) -> Dict[str, Any]:
        """Get dataset information from get_data.py"""
        data_mean = [0.485, 0.456, 0.406]
        data_std = [0.229, 0.224, 0.225]
        
        if not self.normalize:
            data_mean = [0.0, 0.0, 0.0]
            data_std = [1.0, 1.0, 1.0]
            
        # Load ImageNet class names from json file
        import json
        try:
            with open("imagenet_class_index.json", "r") as f:
                class_names = json.load(f)
        except:
            class_names = None
            
        return {
            "data_set": "ImageNet",
            "data_shape": [3, self.input_size, self.input_size],
            "n_bits": 8,
            "temp": 1,
            "num_classes": 1000,
            "class_names": class_names,
            "data_mean": data_mean,
            "data_std": data_std,
        }
        
    def setup(self):
        """Set up the datasets"""
        transforms_list = [
            transforms.Resize(256),
            transforms.CenterCrop(self.input_size),
            transforms.ToTensor(),
        ]
        
        if self.normalize:
            transforms_list.append(
                transforms.Normalize(
                    mean=self.info["data_mean"],
                    std=self.info["data_std"]
                )
            )
            
        transform = transforms.Compose(transforms_list)
        
        self.train_dataset = torchvision.datasets.ImageFolder(
            root=os.path.join(self.data_dir, "train"),
            transform=transform
        )
        
        self.val_dataset = torchvision.datasets.ImageFolder(
            root=os.path.join(self.data_dir, "val"),
            transform=transform
        ) 