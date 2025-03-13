import argparse
import datetime
import math
import os
import random

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torcheval.metrics.functional import binary_f1_score
from torchmetrics.classification import BinaryStatScores
from torchvision import datasets, models, transforms
from tqdm.auto import tqdm

from data.datasets import get_dataset, HiddenDataset
from data.utils import split_dataset
from models import get_prober
from utils.metrics import get_labels_and_onehot, calc_metrics


@torch.no_grad()
def eval(args, data_loader, prober, criterion):
    """Evaluate prober model"""
    prober.eval()
    metric = BinaryStatScores().to(args.device)
    pred_list = []
    correct_list = []
    running_loss = 0.0

    for i, (idx, hiddens, corrects, _, _) in enumerate(data_loader):
        hiddens = hiddens.to(args.device)
        corrects = corrects.to(args.device)

        output = prober(hiddens)
        preds = output.argmax(dim=1)
        running_loss += criterion(output, corrects).item()

        pred_list.append(preds)
        correct_list.append(corrects)

    pred_list = torch.cat(pred_list)
    correct_list = torch.cat(correct_list)

    # Calculate metrics
    acc = (pred_list == correct_list).sum().float() / len(pred_list)
    f1 = binary_f1_score(pred_list, correct_list)
    stats = metric(pred_list, correct_list)

    print(f"\n ========= ⭐️ eval acc: {acc:.4f} / f1: {f1:.4f} ⭐️ ========= ")
    print(f"Stats: {stats}")

    return running_loss / len(data_loader), acc, f1, pred_list, correct_list


def train(args, train_loader, val_loader, prober, train_set, val_set):
    """Train prober model"""
    criterion = nn.CrossEntropyLoss(
        weight=torch.Tensor([args.loss_weight, 1]).to(args.device),
        label_smoothing=args.label_smoothing
    )
    optimizer = torch.optim.Adam(prober.parameters(), lr=args.lr)
    
    train_label, train_onehot = get_labels_and_onehot(train_set)
    val_label, val_onehot = get_labels_and_onehot(val_set)

    for epoch in range(args.epochs):
        # Train
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

            train_loss += loss.item()

        train_loss = train_loss / len(train_loader)
        print(f"\n ========= ⭐️ train (epoch-{epoch}) loss: {train_loss:.4f} ⭐️ ========= ")

        # Evaluate
        train_metrics = calc_metrics(args, train_loader, train_label, train_onehot, prober, criterion)
        val_metrics = calc_metrics(args, val_loader, val_label, val_onehot, prober, criterion)

        print(epoch, args.lr, val_metrics['acc'], val_metrics['f1'])
        # Save checkpoint
        save_path = os.path.join(
            args.save_dir,
            f"prober_ep-{epoch:02d}_lr-{args.lr}_acc-{val_metrics['acc']:.4f}_f1-{val_metrics['f1']:.4f}.pth"
        )
        torch.save(prober.state_dict(), save_path)

        # Early stopping
        if val_metrics['fpr'] > 90 and epoch > 30:
            print(f"Early stopping at epoch {epoch}, FPR: {val_metrics['fpr']:.4f}")
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
    parser.add_argument("--train-ratio", type=float, default=0.8, required=False)
    parser.add_argument("--label-smoothing", type=float, default=0.2)
    parser.add_argument("--loss-weight", type=float, default=4)
    parser.add_argument("--latent-dim1", type=int, default=256)
    parser.add_argument("--latent-dim2", type=int, default=128)
    parser.add_argument("--latent-dim3", type=int, default=64)
    parser.add_argument("--split", type=str, default="add")
    args = parser.parse_args()
    

    updates = {"latent_dim": [args.latent_dim1, args.latent_dim2, args.latent_dim3]}
    args.__dict__.update(updates)

    os.makedirs(args.save_dir, exist_ok=False)


    random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    train_set = get_dataset('hidden', hidden_data_path=args.train_path)
    mean, std = train_set.mean, train_set.std
    val_set = get_dataset('hidden', hidden_data_path=args.valid_path, mean=mean, std=std)
    

    prober_train_set, prober_val_set, split_dict = split_dataset(
        train_set, val_set, args.train_ratio, args.save_dir, args.split
    )

    train_loader = DataLoader(
        prober_train_set, batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(prober_val_set, batch_size=args.batch_size, shuffle=False)

    print(f"Train : {prober_train_set.__len__()}, valid : {prober_val_set.__len__()}")

    prober = get_prober(
        dataset=args.dataset,
        input_dim=args.latent_dim1,
        hidden_dims=[args.latent_dim2, args.latent_dim3]
    ).to(args.device)
    print(prober)


    train(args, train_loader, val_loader, prober, prober_train_set, prober_val_set)


if __name__ == "__main__":
    main()
