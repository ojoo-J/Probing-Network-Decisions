import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List
import numpy as np
import sklearn.metrics

@torch.no_grad()
def accuracy(output: torch.Tensor, target: torch.Tensor, topk: Tuple[int] = (1,)) -> List[torch.Tensor]:
    """Computes the accuracy over the k top predictions"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    correct = pred.eq(target.view(-1, 1).expand_as(pred)) # (batch_size, top_k)
    correct = correct.any(dim=1).float()  # Aggregate into (batch_size)

    acc = correct / len(correct)
    return acc, correct

def calc_entropy(input_tensor):
    """Calculate entropy of input probabilities"""
    lsm = nn.LogSoftmax(dim=1)
    log_probs = lsm(input_tensor)
    probs = torch.exp(log_probs)
    p_log_p = log_probs * probs
    entropy = -p_log_p.mean(dim=1)
    return entropy

def get_labels_and_onehot(dset):
    """Return labels and one-hot encoded vectors from the dataset"""
    onehot_mat = np.eye(2)

    if isinstance(dset, torch.utils.data.ConcatDataset):
        label = []
        onehot = []
        for ds in dset.datasets:
            if isinstance(ds, torch.utils.data.Subset):
                label.append(ds.dataset.data["correct"][ds.indices])
                onehot.append(onehot_mat[ds.dataset.data["correct"][ds.indices]])
            else:
                label.append(ds.data["correct"])
                onehot.append(onehot_mat[dset.data["correct"]])
        label = np.concatenate(label)
        onehot = np.concatenate(onehot)
    elif isinstance(dset, torch.utils.data.Subset):
        label = dset.dataset.data["correct"][dset.indices]
        onehot = onehot_mat[dset.dataset.data["correct"][dset.indices]]
    elif isinstance(dset, torch.utils.data.Dataset):
        label = dset.data["correct"]
        onehot = onehot_mat[dset.data["correct"]]
    else:
        raise ValueError(f"Unknown dataset type: {type(dset)}")

    return label, onehot

@torch.no_grad()
def calculate_metrics(loader, model, device='cuda', in_type='image', out_type='correct'):
    """Calculate basic metrics from model predictions"""
    model.eval()
    total = 0
    correct = 0
    all_preds = []
    all_targets = []

    for batch in loader:
        # Handle different batch formats
        if isinstance(batch, tuple):
            if len(batch) == 5:  # DataItem(index, hidden, correct, img, label)
                x = getattr(batch, in_type)
                target = getattr(batch, out_type)
            elif len(batch) == 3:  # (idx, img, label)
                _, x, target = batch
            else:  # (img, label)
                x, target = batch
        else:  
            raise ValueError(f"Unknown batch format: {type(batch)}")
        
        x = x.to(device)
        target = target.to(device)
        
        # Forward pass
        output = model(x)
        pred = output.argmax(dim=1)
        
        # Accumulate metrics
        total += target.size(0)
        correct += pred.eq(target).sum().item()
        
        all_preds.extend(pred.cpu().numpy())
        all_targets.extend(target.cpu().numpy())

    # Calculate metrics
    accuracy = correct / total
    f1 = sklearn.metrics.f1_score(all_targets, all_preds, average='binary')
    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(all_targets, all_preds).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    return {
        "acc": accuracy,
        "f1": f1, 
        "fpr": fpr
    }

@torch.no_grad()
def get_metric_values(loader, model, device='cuda'):
    """Calculate basic metrics from model predictions"""
    model.eval()
    total = 0
    correct = 0
    all_preds = []
    all_targets = []

    for batch in loader:
        # Handle different batch formats
        if len(batch) == 5:  # (idx, hidden, correct, img, label)
            _, x, target = batch[:3]
        elif len(batch) == 3:  # (idx, img, label)
            _, x, target = batch
        else:  # (img, label)
            x, target = batch
        
        x = x.to(device)
        target = target.to(device)
        
        # Forward pass
        output = model(x)
        pred = output.argmax(dim=1)
        
        # Accumulate metrics
        total += target.size(0)
        correct += pred.eq(target).sum().item()
        
        all_preds.extend(pred.cpu().numpy())
        all_targets.extend(target.cpu().numpy())

    # Calculate accuracy
    accuracy = correct / total

    # Calculate F1 score
    f1 = sklearn.metrics.f1_score(all_targets, all_preds, average='binary')

    # Calculate False Positive Rate (FPR)
    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(all_targets, all_preds).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fpr = fpr

    return {
        "acc": accuracy,
        "f1": f1,
        "fpr": fpr
    }