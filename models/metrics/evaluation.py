import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics
from .losses import ECELoss

def calc_metrics_from_loader(args, dset, loader, prober, criterion, debug=False):
    """Calculate all metrics from a data loader"""
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
                onehot.append(onehot_mat[ds.data["correct"]])
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

    print(label.shape, onehot.shape)

    metric_vals = calc_metrics(args, loader, label, onehot, prober, criterion)
    metric_vals = {
        "acc": metric_vals[0],
        "auroc": metric_vals[1],
        "aupr_success": metric_vals[2],
        "aupr": metric_vals[3],
        "fpr": metric_vals[4],
        "tnr": metric_vals[5],
        "aurc": metric_vals[6],
        "eaurc": metric_vals[7],
        "ece": metric_vals[8],
        "nll": metric_vals[9],
        "brier": metric_vals[10],
    }
    if debug:
        for key, val in metric_vals.items():
            print(f"{key} : {val:.4f}")
    return metric_vals

# ... rest of the evaluation functions ... 