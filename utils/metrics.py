import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List
import numpy as np
from enum import Enum
from sklearn import metrics

def accuracy(output: torch.Tensor, target: torch.Tensor, topk: Tuple[int] = (1,)) -> List[torch.Tensor]:
    """Computes the accuracy over the k top predictions"""
    with torch.no_grad():
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
    log_probs = lsm(input_tensor)  # batch, class
    probs = torch.exp(log_probs)  # batch, class
    p_log_p = log_probs * probs  # batch, class
    entropy = -p_log_p.mean(dim=1)
    return entropy

class ECELoss(nn.Module):
    """Expected Calibration Error Loss"""
    def __init__(self, n_bins=15):
        super().__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        return ece

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

    return label, onehot

def calc_metrics_from_loader(args, dset, loader, prober, criterion, debug=False):
    """Calculate all metrics from a data loader"""
    label, onehot = get_labels_and_onehot(dset)

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

def calc_metrics(args, loader, label, label_onehot, model, criterion):
    vals = get_metric_values(args, loader, model, criterion)
    acc, softmax, correct, logit = vals[:4]
    conf_correct, conf_wrong = vals[4:6]
    ece_correct, ece_wrong = vals[6:8]
    nll_correct, nll_wrong = vals[8:10]

    # Calculate various metrics
    aurc, eaurc = calc_aurc_eaurc(softmax, correct)
    auroc, aupr_success, aupr, fpr, tnr = calc_fpr_aupr(softmax, correct)
    ece = calc_ece(softmax, label, bins=15)
    nll, brier = calc_nll_brier(softmax, logit, label, label_onehot)
    f1 = metrics.f1_score(label, np.argmax(softmax, axis=1))

    return {
        "acc": acc,
        "auroc": auroc * 100,
        "aupr_success": aupr_success * 100,
        "aupr": aupr * 100,
        "fpr": fpr * 100,
        "tnr": tnr * 100,
        "aurc": aurc * 1000,
        "eaurc": eaurc * 1000,
        "ece": ece * 100,
        "nll": nll * 10,
        "brier": brier * 100,
        "f1": f1
    }

def calc_metrics_plot(args, loader, label, label_onehot, model, criterion):
    vals = get_metric_values(args, loader, model, criterion)
    acc, softmax, correct, logit = vals[:4]
    conf_correct, conf_wrong = vals[4:6]
    ece_correct, ece_wrong = vals[6:8]

    aurc, eaurc = calc_aurc_eaurc(softmax, correct)
    auroc, aupr_success, aupr, fpr, tnr = calc_fpr_aupr(softmax, correct)
    ece = calc_ece(softmax, label, bins=15)

    return (
        acc,
        auroc * 100,
        aurc * 1000,
        fpr * 100,
        ece * 100,
        conf_correct * 100,
        conf_wrong * 100,
        ece_correct * 100,
        ece_wrong * 100,
    )

def calc_aurc_eaurc(softmax, correct):
    """Calculate AURC and EAURC metrics"""
    softmax = np.array(softmax)
    correctness = np.array(correct)
    softmax_max = np.max(softmax, 1)

    sort_values = sorted(zip(softmax_max[:], correctness[:]), key=lambda x: x[0], reverse=True)
    sort_softmax_max, sort_correctness = zip(*sort_values)
    risk_li, coverage_li = coverage_risk(sort_softmax_max, sort_correctness)
    aurc, eaurc = aurc_eaurc(risk_li)

    return aurc, eaurc

def calc_fpr_aupr(softmax, correct):
    """Calculate FPR and AUPR metrics"""
    softmax = np.array(softmax)
    correctness = np.array(correct)
    softmax_max = np.max(softmax, 1)

    fpr, tpr, thresholds = metrics.roc_curve(correctness, softmax_max)
    auroc = metrics.auc(fpr, tpr)
    idx_tpr_95 = np.argmin(np.abs(tpr - 0.95))
    fpr_in_tpr_95 = fpr[idx_tpr_95]
    tnr_in_tpr_95 = 1 - fpr[np.argmax(tpr >= 0.95)]

    precision, recall, thresholds = metrics.precision_recall_curve(correctness, softmax_max)
    aupr_success = metrics.auc(recall, precision)
    aupr_err = metrics.average_precision_score(-1 * correctness + 1, -1 * softmax_max)

    print("AUROC {0:.2f}".format(auroc * 100))
    print("AUPR_Success {0:.2f}".format(aupr_success * 100))
    print("AUPR_Error {0:.2f}".format(aupr_err * 100))
    print("FPR@TPR95 {0:.2f}".format(fpr_in_tpr_95 * 100))
    print("TNR@TPR95 {0:.2f}".format(tnr_in_tpr_95 * 100))

    return auroc, aupr_success, aupr_err, fpr_in_tpr_95, tnr_in_tpr_95

def calc_ece(softmax, label, bins=15):
    """Calculate Expected Calibration Error"""
    bin_boundaries = torch.linspace(0, 1, bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    softmax = torch.tensor(softmax)
    labels = torch.tensor(label)

    softmax_max, predictions = torch.max(softmax, 1)
    correctness = predictions.eq(labels.long())

    ece = torch.zeros(1)

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = softmax_max.gt(bin_lower.item()) * softmax_max.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()

        if prop_in_bin.item() > 0.0:
            accuracy_in_bin = correctness[in_bin].float().mean()
            avg_confidence_in_bin = softmax_max[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    print("ECE {0:.2f} ".format(ece.item() * 100))
    return ece.item()

def calc_nll_brier(softmax, logit, label, label_onehot):
    """Calculate NLL and Brier score"""
    brier_score = np.mean(np.sum((softmax - label_onehot) ** 2, axis=1))

    logit = torch.tensor(logit, dtype=torch.float)
    label = torch.tensor(label, dtype=torch.int)
    logsoftmax = torch.nn.LogSoftmax(dim=1)

    log_softmax = logsoftmax(logit)
    nll = calc_nll(log_softmax, label)

    print("NLL {0:.2f} ".format(nll.item() * 10))
    print("Brier {0:.2f}".format(brier_score * 100))

    return nll.item(), brier_score

def calc_nll(log_softmax, label):
    """Calculate Negative Log Likelihood"""
    out = torch.zeros_like(label, dtype=torch.float)
    for i in range(len(label)):
        out[i] = log_softmax[i][label[i]]
    return -out.sum() / len(out)

def coverage_risk(confidence, correctness):
    """Calculate coverage and risk"""
    risk_list = []
    coverage_list = []
    risk = 0
    for i in range(len(confidence)):
        coverage = (i + 1) / len(confidence)
        coverage_list.append(coverage)

        if correctness[i] == 0:
            risk += 1
        risk_list.append(risk / (i + 1))

    return risk_list, coverage_list

def aurc_eaurc(risk_list):
    """Calculate AURC and EAURC from risk list"""
    r = risk_list[-1]
    risk_coverage_curve_area = 0
    optimal_risk_area = r + (1 - r) * np.log(1 - r)
    
    for risk_value in risk_list:
        risk_coverage_curve_area += risk_value * (1 / len(risk_list))

    aurc = risk_coverage_curve_area
    eaurc = risk_coverage_curve_area - optimal_risk_area

    print("AURC {0:.2f}".format(aurc * 1000))
    print("EAURC {0:.2f}".format(eaurc * 1000))

    return aurc, eaurc

def get_metric_values(args, loader, model, criterion):
    """Get all metric values from model predictions"""
    model.eval()
    with torch.no_grad():
        total_loss = 0.0
        total_acc = 0.0
        accuracy = 0.0

        list_softmax = []
        list_correct = []
        list_logit = []
        logits_list = []
        labels_list = []
        conf = []
        correct = []

        for idx, hidden, target, img, label in loader:
            hidden = hidden.cuda()
            target = target.long().cuda()
            output = model(hidden)
            loss = criterion(output, target).cuda()
            
            logits_list.append(output.detach().cpu())
            labels_list.append(target.cpu())
            total_loss += loss.mean().item()
            
            pred = output.data.max(1, keepdim=True)[1]
            prob, _pred = F.softmax(output, dim=1).max(1)
            
            conf.append(prob.detach().cpu().view(-1).numpy())
            correct.append(_pred.cpu().eq(target.cpu().data.view_as(_pred)).numpy())
            total_acc += pred.eq(target.data.view_as(pred)).sum()

            list_logit.extend(output.cpu().data.numpy())
            list_softmax.extend(F.softmax(output, dim=1).cpu().data.numpy())

            for j in range(len(pred)):
                accuracy += 1 if pred[j] == target[j] else 0
                list_correct.append(1 if pred[j] == target[j] else 0)

        # Calculate additional metrics
        ece_criterion = ECELoss().cuda()
        nll_criterion = nn.CrossEntropyLoss().cuda()

        logits = torch.cat(logits_list)
        labels = torch.cat(labels_list)
        conf = np.concatenate(conf, axis=0)
        correct = np.concatenate(correct, axis=0)
        groud = np.ones_like(correct)
        
        conf_wrong = np.mean(conf[groud ^ correct])
        conf_correct = np.mean(conf[correct])
        ece_wrong = ece_criterion(logits[groud ^ correct], labels[groud ^ correct]).item()
        ece_correct = ece_criterion(logits[correct], labels[correct]).item()
        nll_wrong = nll_criterion(logits[groud ^ correct], labels[groud ^ correct]).item()
        nll_correct = nll_criterion(logits[correct], labels[correct]).item()

        total_loss /= len(loader)
        total_acc = 100.0 * total_acc.item() / len(loader.dataset)
        print(f"Accuracy {total_acc:.2f}")

        list_softmax = np.array(list_softmax)
        list_logit = np.array(list_logit)
        list_correct = np.array(list_correct)

    return (
        total_acc,
        list_softmax,
        list_correct,
        list_logit,
        conf_correct,
        conf_wrong,
        ece_correct,
        ece_wrong,
        nll_correct,
        nll_wrong,
    )

# Add other metric calculation functions as needed... 