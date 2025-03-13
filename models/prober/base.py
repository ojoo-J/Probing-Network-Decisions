from ..base import BaseModel
import torch.nn as nn
import torch

class BaseProber(BaseModel):
    """Base class for all probers"""
    def forward(self, x):
        x = x.float()
        output = self.layers(x.view(x.size(0), -1))
        return output

    @torch.no_grad()
    def evaluate_img2correct(self, dataloader, device):
        self.eval()

        train_x = []
        train_y = []
        pred_y = []
        for x, y in dataloader:
            train_x.append(x)
            train_y.append(y)

            pred = self(x.to(device))
            _, pred_class = pred.max(dim=1)
            pred_y.append(pred_class.detach().cpu())

        train_x = torch.cat(train_x)
        train_y = torch.cat(train_y)
        pred_y = torch.cat(pred_y)

        acc = (train_y == pred_y).sum() / len(pred_y)
        print(f"Acc : {acc:.4f}")
        return train_x, train_y, pred_y, acc 