import argparse
import datetime
import math
import os
import random

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torcheval.metrics.functional import binary_f1_score
from torchmetrics.classification import BinaryStatScores
from torchvision import datasets, models, transforms
from tqdm.auto import tqdm

# from transformers import get_scheduler
from utils.compute_metrics import calc_metrics_from_loader
from utils.get_data import HiddenDataset2, split_dataset
from utils.get_prober import *

import wandb


class CosineAnnealingWarmUpRestarts(_LRScheduler):
    def __init__(
        self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1.0, last_epoch=-1
    ):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [
                (self.eta_max - base_lr) * self.T_cur / self.T_up + base_lr
                for base_lr in self.base_lrs
            ]
        else:
            return [
                base_lr
                + (self.eta_max - base_lr)
                * (
                    1
                    + math.cos(
                        math.pi * (self.T_cur - self.T_up) / (self.T_i - self.T_up)
                    )
                )
                / 2
                for base_lr in self.base_lrs
            ]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(
                        math.log(
                            (epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult
                        )
                    )
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult**n - 1) / (
                        self.T_mult - 1
                    )
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch

        self.eta_max = self.base_eta_max * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr


@torch.no_grad()
def eval(args, data_loader, prober, criterion):
    prober.eval()
    metric = BinaryStatScores().to(args.device)
    pred_list = []
    correct_list = []
    logit_list = []
    running_loss = 0.0
    # for i, (idx, hiddens, corrects) in enumerate(tqdm(data_loader)):
    for i, (idx, hiddens, corrects, _, _) in enumerate(data_loader):
        hiddens = hiddens.to(args.device)
        corrects = corrects.to(args.device)

        output = prober(hiddens)
        preds = output.argmax(dim=1)
        running_loss += criterion(output, corrects).item()

        pred_list.append(preds)
        correct_list.append(corrects)
        logit_list.append(preds)

    # print(print(f'\n ========= ⭐️ val (epoch-{e}) acc: {acc}% / f1: {f1}% ⭐️ ========= '))

    pred_list = torch.cat(pred_list)
    correct_list = torch.cat(correct_list)
    logit_list = torch.cat(logit_list)

    acc = (pred_list == correct_list).sum().detach().cpu().numpy() / len(pred_list)
    f1 = binary_f1_score(pred_list, correct_list, threshold=0.5).detach().cpu().numpy()

    print(
        f"\n ========= ⭐️ eval acc: {acc:.4f}% / f1: {f1:.4f}%⭐️ / {torch.unique(pred_list).detach().cpu().numpy()}========= "
    )
    stats = metric(pred_list, correct_list).detach().cpu().numpy()
    print(
        f"#pos, #neg : {(correct_list == 1).detach().cpu().sum().item()} \t {(correct_list == 0).detach().cpu().sum().item()}"
    )
    print(f"tp, fp, tn, fn, sup : {stats}")

    return running_loss, acc, f1, pred_list, correct_list


def train(args, train_loader, val_loader, prober, train_set, val_set):
    criterion = nn.CrossEntropyLoss(
        weight=torch.Tensor([args.loss_weight, 1]),
        label_smoothing=args.label_smoothing,
    ).to(
        args.device
    )  # weight=torch.Tensor([1,9])
    optimizer = torch.optim.Adam(prober.parameters(), lr=args.lr)

    metric_list = []

    for e in range(args.epochs):
        prober.train()
        train_loss = 0
        for i, (idx, hiddens, corrects, _, _) in enumerate(tqdm(train_loader)):
            hiddens = hiddens.to(args.device)
            corrects = corrects.to(args.device)

            output = prober(hiddens)
            loss = criterion(output, corrects)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # lr_scheduler.step()

            train_loss += loss.item()

        total_loss = round(train_loss / len(train_loader), 4)
        print(f"\n ========= ⭐️ train (epoch-{e}) loss: {total_loss} ⭐️ ========= ")

        train_loss, train_acc, train_f1, _, _ = eval(
            args, train_loader, prober, criterion
        )
        val_loss, val_acc, val_f1, pred_list, correct_list = eval(
            args, val_loader, prober, criterion
        )

        metric_vals = dict()
        metric_vals["train_loss"] = total_loss
        metric_vals["train_f1"] = train_f1
        metric_vals["valid_loss"] = val_loss
        metric_vals["valid_f1"] = val_f1

        tmp_metric_vals = calc_metrics_from_loader(
            args, train_set, train_loader, prober, criterion
        )
        for key, val in tmp_metric_vals.items():
            metric_vals[f"train_{key}"] = val

        tmp_metric_vals = calc_metrics_from_loader(
            args, val_set, val_loader, prober, criterion
        )
        for key, val in tmp_metric_vals.items():
            metric_vals[f"valid_{key}"] = val

        wandb.log(metric_vals, step=e)

        torch.save(
            prober.state_dict(),
            os.path.join(
                args.save_dir,
                f"prober_ep-{e:02d}_lr-{args.lr}_acc-{val_acc:.4f}_f1-{val_f1:.4f}.pth",
            ),
        )

        if (metric_vals["valid_fpr"] > 90) and (e > 30):
            print(
                f"early stop due to valid_fpr > 90 after epoch {e}: {metric_vals['valid_fpr']:.4f}"
            )
            print(metric_vals)
            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="MNIST")
    parser.add_argument(
        "--train-path",
        type=str,
        default="/project/run/outputs/<classifier path for train dataset>",
    )
    parser.add_argument(
        "--valid-path",
        type=str,
        default="/project/outputs/<classifier path for val dataset>",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="/project/outputs/",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--label-smoothing", type=float, default=0.2)
    parser.add_argument("--loss-weight", type=float, default=4)
    parser.add_argument("--latent-dim1", type=int, default=256)
    parser.add_argument("--latent-dim2", type=int, default=128)
    parser.add_argument("--latent-dim3", type=int, default=64)
    parser.add_argument("--split", type=str, default="add")
    args = parser.parse_args()
    
    run = wandb.init(
        # project=f"train_prober_sweep",
        project=f"train_prober_{args.dataset}",
    )

    updates = {"latent_dim": [args.latent_dim1, args.latent_dim2, args.latent_dim3]}
    args.__dict__.update(updates)

    now = datetime.datetime.now()
    save_dir = os.path.join(args.save_dir, now.strftime("%Y-%m-%d_%H%M%S"))
    args.save_dir = save_dir
    os.makedirs(save_dir, exist_ok=True)

    wandb.config.update(args, allow_val_change=True)
    print(wandb.config)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    train_set = HiddenDataset2(args.train_path)
    mean, std = train_set.mean, train_set.std
    val_set = HiddenDataset2(args.valid_path)
    
    train_set.data['hidden'] = (train_set.data['hidden'] - mean) / std
    val_set.data['hidden'] = (val_set.data['hidden'] - mean) / std
    

    prober_train_set, prober_val_set, split_dict = split_dataset(
        train_set, val_set, args.train_ratio, args.save_dir, args.split
    )

    train_loader = DataLoader(
        prober_train_set, batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(prober_val_set, batch_size=args.batch_size, shuffle=False)

    print(f"Train : {prober_train_set.__len__()}, valid : {prober_val_set.__len__()}")

    prober = ProberMNIST(args.latent_dim)
    print(prober)
    prober.to(args.device)

    wandb.watch(prober)

    train(args, train_loader, val_loader, prober, prober_train_set, prober_val_set)


if __name__ == "__main__":
    main()
