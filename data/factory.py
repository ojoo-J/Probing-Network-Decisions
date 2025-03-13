from typing import Optional, Union
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from .hidden import HiddenDataset, split_dataset

def get_dataset(
    name: str,
    root: str = None,
    split: str = 'train',
    transform: Optional[transforms.Compose] = None,
    path: Optional[str] = None,
    train_ratio: float = 0.8,
    seed: int = 42
) -> Union[Dataset, tuple[Dataset, Dataset]]:
    """Get dataset by name"""
    if name == 'hidden':
        if path is None:
            raise ValueError("Path required for hidden dataset")
        dataset = HiddenDataset(path)
        if split == 'split':
            return split_dataset(dataset, train_ratio, seed)
        return dataset
        
    if transform is None:
        if name == 'MNIST':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        elif name in ['CIFAR10', 'CIFAR100']:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465),
                    (0.2023, 0.1994, 0.2010)
                )
            ])
        else:
            raise ValueError(f"No default transform for dataset {name}")

    if name == 'MNIST':
        dataset = datasets.MNIST(
            root=root,
            train=(split == 'train'),
            transform=transform,
            download=True
        )
    elif name == 'CIFAR10':
        dataset = datasets.CIFAR10(
            root=root,
            train=(split == 'train'),
            transform=transform,
            download=True
        )
    elif name == 'CIFAR100':
        dataset = datasets.CIFAR100(
            root=root,
            train=(split == 'train'),
            transform=transform,
            download=True
        )
    else:
        raise ValueError(f"Unknown dataset {name}")

    return dataset 