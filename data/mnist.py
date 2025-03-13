from typing import Dict, Any, Tuple, Optional
import torch
import torchvision
import torchvision.transforms as transforms
from .base import BaseDataModule

class MNISTDataModule(BaseDataModule):
    """MNIST data module"""
    def __init__(
        self, 
        data_dir: str, 
        batch_size: int = 32,
        normalize: bool = False
    ):
        self.normalize = normalize
        super().__init__(data_dir, batch_size)
        
    def _get_data_info(self) -> Dict[str, Any]:
        """Get dataset information from get_data.py"""
        data_mean = [0.1307]
        data_std = [0.3081]
        
        if not self.normalize:
            data_mean = [0.0]
            data_std = [1.0]
            
        return {
            "data_set": "MNIST",
            "data_shape": [1, 28, 28],
            "n_bits": 8,
            "temp": 1,
            "num_classes": 10,
            "class_names": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
            "data_mean": data_mean,
            "data_std": data_std,
        }
        
    def setup(self):
        """Set up the datasets"""
        transforms_list = [transforms.ToTensor()]
        
        if self.normalize:
            transforms_list.append(
                transforms.Normalize(
                    mean=self.info["data_mean"],
                    std=self.info["data_std"]
                )
            )
            
        transform = transforms.Compose(transforms_list)
        
        self.train_dataset = torchvision.datasets.MNIST(
            root=self.data_dir,
            train=True,
            download=True,
            transform=transform
        )
        
        self.val_dataset = torchvision.datasets.MNIST(
            root=self.data_dir,
            train=False,
            download=True,
            transform=transform
        )


class FashionMNISTDataModule(MNISTDataModule):
    """FashionMNIST data module"""
    def _get_data_info(self) -> Dict[str, Any]:
        """Get dataset information from get_data.py"""
        data_mean = [0.2860]
        data_std = [0.3530]
        
        if not self.normalize:
            data_mean = [0.0]
            data_std = [1.0]
            
        return {
            "data_set": "FashionMNIST",
            "data_shape": [1, 28, 28],
            "n_bits": 8,
            "temp": 1,
            "num_classes": 10,
            "class_names": ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                          'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'],
            "data_mean": data_mean,
            "data_std": data_std,
        }
        
    def setup(self):
        """Set up the datasets"""
        transforms_list = [transforms.ToTensor()]
        
        if self.normalize:
            transforms_list.append(
                transforms.Normalize(
                    mean=self.info["data_mean"],
                    std=self.info["data_std"]
                )
            )
            
        transform = transforms.Compose(transforms_list)
        
        self.train_dataset = torchvision.datasets.FashionMNIST(
            root=self.data_dir,
            train=True,
            download=True,
            transform=transform
        )
        
        self.val_dataset = torchvision.datasets.FashionMNIST(
            root=self.data_dir,
            train=False,
            download=True,
            transform=transform
        ) 