import os
import json
import numpy as np
from collections import defaultdict
from torch.utils.data import random_split
import torch.utils.data

def split_dataset(train_set, val_set, train_ratio, save_dir, split="add"):
    # MNIST original split train : valid = 60000 : 10000
    # CIFAR10 original split train : valid = 50000 : 10000
    """
    # Split options:
    # - "add": Uses validation set data for both training and validation
    #   * Takes portion of validation set and adds it to training set
    #   * Remaining validation data used for probing validation
    #   * Results in overlapping data between classifier and prober training
    #
    # - "disjoint": Keeps classifier and prober data separate
    #   * Uses original training data only for classifier
    #   * Splits validation set into prober train/val sets
    #   * Ensures no overlap between classifier and prober data
    #
    # - "clf": Uses same splits as original classifier
    #   * Training data used for both classifier and prober training
    #   * Validation data used for both classifier and prober validation
    #   * Maintains original dataset splits
    """
    
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

    elif split == "clf":
        prober_train_set = train_set
        prober_val_set = val_set
        
        split_dict = defaultdict(list)
        split_dict["train"] = train_set.data["index"]
        split_dict["test"] = val_set.data["index"]
        split_dict["prober_train"] = train_set.data["index"]
        split_dict["prober_valid"] = val_set.data["index"]
        
    else:
        assert False, "split should be one of add, disjoint, or clf"
    
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