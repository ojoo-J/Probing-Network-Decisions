import json
import os
import pickle
import random
from collections import defaultdict
from typing import Dict

import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, random_split


class HiddenDataset(Dataset):
    def __init__(self, correct_data_path, wrong_data_path):
        with open(correct_data_path, "rb") as fr:
            self.correct_hidden = pickle.load(fr)
        with open(wrong_data_path, "rb") as fr:
            self.wrong_hidden = pickle.load(fr)

        self.data = []
        for hidden in tqdm(self.correct_hidden):
            for i in range(hidden.shape[0]):
                data_dict = {}
                data_dict["hidden"] = hidden[i].cpu().detach()
                data_dict["label"] = 1
                self.data.append(data_dict)

        for hidden in tqdm(self.wrong_hidden):
            for i in range(hidden.shape[0]):
                data_dict = {}
                data_dict["hidden"] = hidden[i].cpu().detach()
                data_dict["label"] = 0
                self.data.append(data_dict)

        random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        hidden = self.data[idx]["hidden"]
        label = self.data[idx]["label"]
        return hidden, label


class HiddenDataset2(Dataset):
    def __init__(self, hidden_data_path, mean=None, std=None):
        self.data_path = hidden_data_path
        with open(self.data_path, "rb") as fr:
            self.data = pickle.load(fr)

        if (mean is not None) and (std is not None): # Test
            self.mean = mean
            self.std = std
        elif (mean is None) and (std is None): # Train
            self.mean, self.std = self.get_stat() 
        else:
            assert False, "mean and std should be both None or not None"
            
        print (self.mean, self.std)
        # for idx in range(len(self.data)):
        #     self.data["hidden"][idx] = (
        #         self.data["hidden"][idx] - self.mean
        #     ) / self.std

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
        index = self.data["index"][idx]
        img = self.data["image"][idx]
        label = self.data["label"][idx]
        hidden = self.data["hidden"][idx]
        correct = self.data["correct"][idx]
        return index, hidden, correct, img, label

    def get_stat(self):
        mean, std = self.data["hidden"].mean(), self.data["hidden"].std()
        if isinstance(mean, torch.Tensor):
            mean = mean.item()
            std = std.item()
        print(f"Data path: {self.data_path}, mean: {mean:.4f}, std: {std:.4f}")
        return mean, std


def get_data_info(dataset: str, normalize: bool = False) -> Dict:
    """
    returns information (class names, image shape, ...) about data set as a dictionary
    """
    n_bits = 8
    temp = 1
    num_classes = 10
    class_names = None
    if dataset == "EMNIST":
        data_shape = [1, 28, 28]
        
    elif dataset == "FashionMNIST":
        data_shape = [1, 28, 28]
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        
        data_mean = np.array([0.2860])
        data_std = np.array([0.3530])
        
    elif dataset == "MNIST":
        data_shape = [1, 28, 28]
        class_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        data_mean = np.array([0.1307])
        data_std = np.array([0.3081])

    elif dataset == "CIFAR10":
        class_names = [
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
        ]
        num_classes = 10
        data_shape = [3, 32, 32]

        data_mean = np.array([0.4914, 0.4822, 0.4465])
        data_std = np.array([0.2023, 0.1994, 0.2010])

    elif dataset == "CIFAR100":
        class_names = [
            "apple",
            "aquarium_fish",
            "baby",
            "bear",
            "beaver",
            "bed",
            "bee",
            "beetle",
            "bicycle",
            "bottle",
            "bowl",
            "boy",
            "bridge",
            "bus",
            "butterfly",
            "camel",
            "can",
            "castle",
            "caterpillar",
            "cattle",
            "chair",
            "chimpanzee",
            "clock",
            "cloud",
            "cockroach",
            "couch",
            "crab",
            "crocodile",
            "cup",
            "dinosaur",
            "dolphin",
            "elephant",
            "flatfish",
            "forest",
            "fox",
            "girl",
            "hamster",
            "house",
            "kangaroo",
            "keyboard",
            "lamp",
            "lawn_mower",
            "leopard",
            "lion",
            "lizard",
            "lobster",
            "man",
            "maple_tree",
            "motorcycle",
            "mountain",
            "mouse",
            "mushroom",
            "oak_tree",
            "orange",
            "orchid",
            "otter",
            "palm_tree",
            "pear",
            "pickup_truck",
            "pine_tree",
            "plain",
            "plate",
            "poppy",
            "porcupine",
            "possum",
            "rabbit",
            "raccoon",
            "ray",
            "road",
            "rocket",
            "rose",
            "sea",
            "seal",
            "shark",
            "shrew",
            "skunk",
            "skyscraper",
            "snail",
            "snake",
            "spider",
            "squirrel",
            "streetcar",
            "sunflower",
            "sweet_pepper",
            "table",
            "tank",
            "telephone",
            "television",
            "tiger",
            "tractor",
            "train",
            "trout",
            "tulip",
            "turtle",
            "wardrobe",
            "whale",
            "willow_tree",
            "wolf",
            "woman",
            "worm",
        ]
        num_classes = 100
        data_shape = [3, 32, 32]

        data_mean = np.array([0.5071, 0.4867, 0.4408])
        data_std = np.array([0.2675, 0.2565, 0.2761])

    elif dataset == "ImageNette":
        class_names = [
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
        ]
        num_classes = 10
        data_shape = [3, 128, 128]

        # Just follow imagenet stat
        data_mean = np.array([0.485, 0.456, 0.406])
        data_std = np.array([0.229, 0.224, 0.225])

    elif dataset == "ImageNet":
        import json

        with open("imagenet_class_index.json", "r") as f:
            class_names = json.load(f)
        num_classes = 1000
        data_shape = [3, 224, 224]

        data_mean = np.array([0.485, 0.456, 0.406])
        data_std = np.array([0.229, 0.224, 0.225])

    if not normalize:
        data_mean = np.zeros_like(data_mean)
        data_std = np.ones_like(data_std)

    data_info = {
        "data_set": dataset,
        "data_shape": data_shape,
        "n_bits": n_bits,
        "temp": temp,
        "num_classes": num_classes,
        "class_names": class_names,
        "data_mean": data_mean,
        "data_std": data_std,
    }
    return data_info


def get_dataset(dataset, data_dir):
    data_info = get_data_info(dataset, normalize=True)
    if dataset == "FashionMNIST":
        train_dataset = torchvision.datasets.FashionMNIST(
            root=data_dir,
            train=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                ]
            ),)
        val_dataset = torchvision.datasets.FashionMNIST(
            root=data_dir,
            train=False,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                ]
            ),
        )
    elif dataset == "MNIST":
        train_dataset = torchvision.datasets.MNIST(
            root=data_dir,
            train=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                ]
            ),
        )
        val_dataset = torchvision.datasets.MNIST(
            root=data_dir,
            train=False,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                ]
            ),
        )
    elif dataset == "CIFAR10":
        train_dataset = torchvision.datasets.CIFAR10(
            root=data_dir,
            train=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(
                        data_info["data_mean"], data_info["data_std"]
                    ),
                ]
            ),
        )

        val_dataset = torchvision.datasets.CIFAR10(
            root=data_dir,
            train=False,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(
                        data_info["data_mean"], data_info["data_std"]
                    ),
                ]
            ),
        )

    elif dataset == "CIFAR100":
        train_dataset = torchvision.datasets.CIFAR100(
            root=data_dir,
            train=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(
                        data_info["data_mean"], data_info["data_std"]
                    ),
                ]
            ),
            download=True,
        )

        val_dataset = torchvision.datasets.CIFAR100(
            root=data_dir,
            train=False,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(
                        data_info["data_mean"], data_info["data_std"]
                    ),
                ]
            ),
            download=True,
        )

    elif dataset == "ImageNette":
        # This is setting for inference, not training
        normalize = torchvision.transforms.Normalize(
            mean=data_info["data_mean"], std=data_info["data_std"]
        )
        ts = torchvision.transforms.Compose(
            [
                # torchvision.transforms.Resize(160),
                torchvision.transforms.CenterCrop(128),
                torchvision.transforms.ToTensor(),
                normalize,
            ]
        )
        train_dataset = torchvision.datasets.ImageFolder(
            root=os.path.join(data_dir, "train"), transform=ts
        )
        val_dataset = torchvision.datasets.ImageFolder(
            root=os.path.join(data_dir, "val"), transform=ts
        )

    elif dataset == "ImageNet":
        normalize = torchvision.transforms.Normalize(
            mean=data_info["data_mean"], std=data_info["data_std"]
        )
        ts = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(256),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                normalize,
            ]
        )
        train_dataset = torchvision.datasets.ImageFolder(
            root=os.path.join(data_dir, "train"), transform=ts
        )
        val_dataset = torchvision.datasets.ImageFolder(
            root=os.path.join(data_dir, "val"), transform=ts
        )

    return train_dataset, val_dataset


def split_dataset(train_set, val_set, train_ratio, save_dir, split="add"):
    # MNIST original split train : valid = 60000 : 10000
    ### Change to train : valid = 6 + 0.8 : 0.2

    # CIFAR10 original split train : valid = 50000 : 10000
    ### Change to train : valid = 6 + 0.8 : 0.2 
    
    if split == "add":
        add_train_size = int(train_ratio * len(val_set))
        val_size = len(val_set) - add_train_size

        add_train_set, prober_val_set = random_split(val_set, [add_train_size, val_size])
        prober_train_set = torch.utils.data.ConcatDataset([train_set, add_train_set])
        
        split_dict = defaultdict(list)
        split_dict["train"] = train_set.data["index"]
        split_dict["test"] = (
            np.array(val_set.data["index"])[prober_val_set.indices]
        ).tolist()
        split_dict["inter"] = sorted(set(val_set.data["index"]) - set(split_dict["test"]))
        split_dict["prober_train"] = sorted(
            set(split_dict["train"]).union(set(split_dict["inter"]))
        )
        split_dict["prober_valid"] = split_dict["test"]
        
    elif split == "disjoint":
        train_size = int(train_ratio * len(val_set))
        val_size = len(val_set) - train_size
        prober_train_set, prober_val_set = random_split(val_set, [train_size, val_size])
        
        split_dict = defaultdict(list)
        split_dict["train"] = train_set.data["index"]
        split_dict["test"] = val_set.data["index"]
        split_dict["prober_train"] = (
            np.array(val_set.data["index"])[prober_train_set.indices]
        ).tolist()
        split_dict["prober_valid"] = (
            np.array(val_set.data["index"])[prober_val_set.indices]
        ).tolist()

    elif split == "cls":
        prober_train_set = train_set
        prober_val_set = val_set
        
        split_dict = defaultdict(list)
        split_dict["train"] = train_set.data["index"]
        split_dict["test"] = val_set.data["index"]
        split_dict["prober_train"] = train_set.data["index"]
        split_dict["prober_valid"] = val_set.data["index"]
        
    else:
        assert False, "split should be one of add, disjoint, or cls"
    
    with open(os.path.join(save_dir, "split.json"), "w") as fw:
        json.dump(split_dict, fw, indent=2)

    return prober_train_set, prober_val_set, split_dict


def get_correct_labels(class_labels, class_pred):
    return (class_labels == class_pred).long()


def get_index_tn(correct_labels, correct_preds):
    """
    index of true negative (TN)
    TN = classifier predicts wrong class and prober is correct
    Samples where prober knows that the classifier is wrong
    """
    indices = torch.where((correct_labels == 0) & (correct_preds == 0))
    indices = indices[0]
    if indices.is_cuda:
        indices = indices.detach().cpu()
    indices = indices.numpy()
    return indices
