import pickle
import torch
from torch.utils.data import Dataset, random_split
import numpy as np
from collections import defaultdict
import os
import json

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

class HiddenDataset(Dataset):
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