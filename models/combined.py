import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from .base import BaseModel
from utils.compute_metrics import calc_entropy

class CombinedNN(BaseModel):
    """Combines classifier and prober models"""
    def __init__(self, classifier, prober, hidden_mean, hidden_std):
        super().__init__()
        self.classifier = classifier
        self.prober = prober
        self.hidden_mean = hidden_mean
        self.hidden_std = hidden_std

        self.freeze(self.prober)
        self.freeze(self.classifier)

    def freeze(self, model):
        for param in model.parameters():
            param.requires_grad = False

    def forward(self, x):
        hidden = self.classifier(x)
        hidden = (hidden - self.hidden_mean) / self.hidden_std
        correct = self.prober(hidden)
        return correct
        
    @torch.no_grad()
    def evaluate_all(self, data_loader, device):
        vals = defaultdict(list)
        for img, label in data_loader:
            img = img.to(device)
            label = torch.Tensor([int(l) for l in label]).to(device)

            cls_out = self.classifier(img)
            prober_out = self(img)

            cls_prob = F.softmax(cls_out, dim=1)
            prober_prob = F.softmax(prober_out, dim=1)

            cls_pred = torch.argmax(cls_prob, dim=1)
            prober_pred = torch.argmax(prober_prob, dim=1)

            hidden = self(img)
            correct = cls_pred == label

            vals["image"].append(img)
            vals["label"].append(label)
            vals["hidden"].append(hidden)
            vals["correct"].append(correct)

            vals["clf_out"].append(cls_out)
            vals["prb_out"].append(prober_out)

            vals["clf_prob"].append(cls_prob)
            vals["prb_prob"].append(prober_prob)

            vals["clf_pred"].append(cls_pred)
            vals["prb_pred"].append(prober_pred)

        for key, val in vals.items():
            vals[key] = torch.cat(val, dim=0).detach().cpu()

        for key in ["clf_prob", "prb_prob"]:
            vals[f"{key}_max"] = vals[key].max(dim=1).values

        vals["clf_prob_max"] = vals["clf_prob"].max(dim=1).values
        vals["clf_entropy"] = calc_entropy(vals["clf_out"])
        return vals


class Identity(BaseModel):
    """Identity model that returns input unchanged"""
    def forward(self, x):
        return x 