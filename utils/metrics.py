import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List
import numpy as np
from sklearn import metrics

@torch.no_grad()
def accuracy(output: torch.Tensor, target: torch.Tensor, topk: Tuple[int] = (1,)) -> List[torch.Tensor]:
    """Computes the accuracy over the k top predictions"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

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

def calc_metrics(args, loader, label, label_onehot, model, criterion):
    """Calculate basic metrics from model predictions"""
    model.eval()
    with torch.no_grad():
        total_acc = 0
        total_f1 = 0
        n_batches = 0

        for idx, hidden, target, img, label in loader:
            hidden = hidden.cuda()
            target = target.long().cuda()
            output = model(hidden)
            
            # Calculate accuracy
            pred = output.data.max(1, keepdim=True)[1]
            total_acc += pred.eq(target.data.view_as(pred)).sum().item()
            
            # Calculate F1 score
            f1 = metrics.f1_score(
                target.cpu().numpy(),
                pred.cpu().numpy(),
                average='binary'
            )
            total_f1 += f1
            n_batches += 1

        avg_acc = (total_acc * 100.0) / len(loader.dataset)
        avg_f1 = (total_f1 * 100.0) / n_batches

        return {
            "acc": avg_acc,
            "f1": avg_f1
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
        if len(batch) == 5:  # (idx, hidden, target, img, label)
            _, x, target = batch[:3]
        elif len(batch) == 3:  # (idx, img, target)
            _, x, target = batch
        else:  # (img, target)
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

    return {
        "accuracy": 100. * correct / total,
        "f1": metrics.f1_score(all_targets, all_preds, average='binary') * 100
    }