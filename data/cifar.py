from typing import Dict, Any, Tuple, Optional
import torch
import torchvision
import torchvision.transforms as transforms
from .base import BaseDataModule

class CIFAR10DataModule(BaseDataModule):
    """CIFAR10 data module"""
    def __init__(
        self, 
        data_dir: str, 
        batch_size: int = 32,
        normalize: bool = False
    ):
        super().__init__(data_dir, batch_size)
        self.normalize = normalize
        
    def _get_data_info(self) -> Dict[str, Any]:
        """Get dataset information from get_data.py"""
        data_mean = [0.4914, 0.4822, 0.4465]
        data_std = [0.2023, 0.1994, 0.2010]
        
        if not self.normalize:
            data_mean = [0.0, 0.0, 0.0]
            data_std = [1.0, 1.0, 1.0]
            
        return {
            "data_set": "CIFAR10",
            "data_shape": [3, 32, 32],
            "n_bits": 8,
            "temp": 1,
            "num_classes": 10,
            "class_names": [
                "airplane",
                "automobile",
                "bird",
                "cat",
                "deer",
                "dog",
                "frog",
                "horse",
                "ship",
                "truck",
            ],
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
        
        self.train_dataset = torchvision.datasets.CIFAR10(
            root=self.data_dir,
            train=True,
            download=True,
            transform=transform
        )
        
        self.val_dataset = torchvision.datasets.CIFAR10(
            root=self.data_dir,
            train=False,
            download=True,
            transform=transform
        )


class CIFAR100DataModule(CIFAR10DataModule):
    """CIFAR100 data module"""
    def _get_data_info(self) -> Dict[str, Any]:
        """Get dataset information from get_data.py"""
        data_mean = [0.5071, 0.4867, 0.4408]
        data_std = [0.2675, 0.2565, 0.2761]
        
        if not self.normalize:
            data_mean = [0.0, 0.0, 0.0]
            data_std = [1.0, 1.0, 1.0]
            
        return {
            "data_set": "CIFAR100",
            "data_shape": [3, 32, 32],
            "n_bits": 8,
            "temp": 1,
            "num_classes": 100,
            "class_names": [
                "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle", 
                "bicycle", "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel", 
                "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock", 
                "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur", 
                "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster", 
                "house", "kangaroo", "keyboard", "lamp", "lawn_mower", "leopard", "lion",
                "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain", "mouse",
                "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear",
                "pickup_truck", "pine_tree", "plain", "plate", "poppy", "porcupine",
                "possum", "rabbit", "raccoon", "ray", "road", "rocket", "rose", "sea",
                "seal", "shark", "shrew", "skunk", "skyscraper", "snail", "snake",
                "spider", "squirrel", "streetcar", "sunflower", "sweet_pepper", "table",
                "tank", "telephone", "television", "tiger", "tractor", "train", "trout",
                "tulip", "turtle", "wardrobe", "whale", "willow_tree", "wolf", "woman",
                "worm"
            ],
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
        
        self.train_dataset = torchvision.datasets.CIFAR100(
            root=self.data_dir,
            train=True,
            download=True,
            transform=transform
        )
        
        self.val_dataset = torchvision.datasets.CIFAR100(
            root=self.data_dir,
            train=False,
            download=True,
            transform=transform
        ) 