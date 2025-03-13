import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional
import pickle
import numpy as np

class BaseDataset:
    def __init__(self, data_dir: str, batch_size: int = 32, normalize: bool = False):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.normalize = normalize
        self.train_dataset = None
        self.val_dataset = None
        self.setup()
    
    def get_transforms(self):
        transforms_list = [transforms.ToTensor()]
        if self.normalize:
            transforms_list.append(
                transforms.Normalize(mean=self.mean, std=self.std)
            )
        return transforms.Compose(transforms_list)
    
    def get_loaders(self) -> Tuple[DataLoader, DataLoader]:
        return (
            DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True),
            DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
        )

class MNIST(BaseDataset):
    mean = [0.1307]
    std = [0.3081]
    
    def setup(self):
        transform = self.get_transforms()
        self.train_dataset = torchvision.datasets.MNIST(
            self.data_dir, train=True, transform=transform, download=True
        )
        self.val_dataset = torchvision.datasets.MNIST(
            self.data_dir, train=False, transform=transform, download=True
        )

class CIFAR10(BaseDataset):
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    
    def setup(self):
        transform = self.get_transforms()
        self.train_dataset = torchvision.datasets.CIFAR10(
            self.data_dir, train=True, transform=transform, download=True
        )
        self.val_dataset = torchvision.datasets.CIFAR10(
            self.data_dir, train=False, transform=transform, download=True
        )

class ImageNet(BaseDataset):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    def setup(self):
        transform = self.get_transforms()
        self.train_dataset = torchvision.datasets.ImageFolder(
            root=f"{self.data_dir}/train",
            transform=transform
        )
        self.val_dataset = torchvision.datasets.ImageFolder(
            root=f"{self.data_dir}/val",
            transform=transform
        )

class HiddenDataset(Dataset):
    def __init__(self, hidden_data_path: str, mean: Optional[float] = None, std: Optional[float] = None):
        self.data_path = hidden_data_path
        with open(self.data_path, "rb") as fr:
            self.data = pickle.load(fr)

        if (mean is not None) and (std is not None):  # Test
            self.mean = mean
            self.std = std
        elif (mean is None) and (std is None):  # Train
            self.mean, self.std = self.get_stat() 
        else:
            raise ValueError("mean and std should be both None or not None")
            
        print(self.mean, self.std)
        
        self.data["correct"] = self.data["correct"].astype(int)
        self.display_stat()

    def display_stat(self):
        print("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
        print(f"Loaded hidden data from {self.data_path}")
        print(f"Number of data: {len(self.data['index'])}")
        print(f"Index: {self.data['index'][:10]} ... {self.data['index'][-10:]}")
        print(f"Image shape: {self.data['image'].shape}")
        print(f"Label shape: {self.data['label'].shape}")
        print(f"Hidden shape: {self.data['hidden'].shape}")
        print(f"Correct label shape: {self.data['correct'].shape}")
        print(f"Mean: {self.mean:.4f}, std: {self.std:.4f}")
        print(
            f'Correct : {(self.data["correct"] == 1).sum()} ({(self.data["correct"] == 1).sum() / len(self.data["correct"]) : .2f}%)'
        )
        print(
            f'Wrong : {(self.data["correct"] == 0).sum()} ({(self.data["correct"] == 0).sum() / len(self.data["correct"]) : .2f}%)'
        )
        print("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")

    def __len__(self):
        return len(self.data["hidden"])

    def __getitem__(self, idx):
        return (
            self.data['index'][idx],
            self.data['hidden'][idx],
            self.data['correct'][idx],
            self.data['image'][idx],
            self.data['label'][idx],
        )
    
    def get_stat(self):
        mean, std = self.data["hidden"].mean(), self.data["hidden"].std()
        if isinstance(mean, torch.Tensor):
            mean = mean.item()
            std = std.item()
        print(f"Data path: {self.data_path}, mean: {mean:.4f}, std: {std:.4f}")
        return mean, std

# Dataset registry
DATASETS = {
    'mnist': MNIST,
    'cifar10': CIFAR10,
    'imagenet': ImageNet,
    'hidden': HiddenDataset,
}

def get_dataset(name: str, **kwargs):
    """Get dataset by name"""
    dataset_class = DATASETS.get(name.lower())
    if dataset_class is None:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASETS.keys())}")
    return dataset_class(**kwargs)



# from data import get_dataset

# # For standard datasets
# mnist_dataset = get_dataset('mnist', data_dir='./data', batch_size=32, normalize=True)

# # For hidden dataset
# hidden_dataset = get_dataset('hidden', hidden_data_path='path/to/hidden/data.pkl')

# def main(args):
#     # Get dataset and loaders
#     dataset = get_dataset(
#         args.dataset,
#         data_dir=args.data_dir,
#         batch_size=args.batch_size,
#         normalize=True
#     )
#     train_loader, val_loader = dataset.get_loaders()
    
#     # Your experiment code...