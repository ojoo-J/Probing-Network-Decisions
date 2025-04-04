import sys
import argparse
import os
import sys
import random
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Add project root to system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.datasets import get_dataset, HiddenDataset
from data.utils import split_dataset
from models import get_prober
from utils.metrics import get_labels_and_onehot, calculate_metrics


def train(args, train_loader, val_loader, prober, train_set, val_set):
    """Train prober model"""
    criterion = nn.CrossEntropyLoss(
        weight=torch.Tensor([args.loss_weight, 1]).to(args.device),
        label_smoothing=args.label_smoothing
    )
    optimizer = torch.optim.Adam(prober.parameters(), lr=args.lr)
    
    train_label, train_onehot = get_labels_and_onehot(train_set)
    val_label, val_onehot = get_labels_and_onehot(val_set)

    pbar = tqdm(range(args.epochs), desc="Train Epochs", position=0, leave=True)
    for epoch in pbar:
        # Train
        prober.train()
        train_loss = 0
        for i, (idx, hiddens, corrects, _, _) in enumerate(train_loader):
            hiddens = hiddens.to(args.device)
            corrects = corrects.to(args.device)

            output = prober(hiddens)
            loss = criterion(output, corrects)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss = train_loss / len(train_loader)
        
        # Evaluate
        train_metrics = calculate_metrics(train_loader, prober, in_type='hidden', out_type='correct')
        val_metrics = calculate_metrics(val_loader, prober, in_type='hidden', out_type='correct')

        pbar.set_postfix({
            "[Train] Acc": f"{train_metrics['acc']:.4f}",
            "F1": f"{train_metrics['f1']:.4f}",
            "FPR": f"{train_metrics['fpr']:.4f}",
            "[Val] Acc": f"{val_metrics['acc']:.4f}",
            "F1": f"{val_metrics['f1']:.4f}",
            "FPR": f"{val_metrics['fpr']:.4f}"
        })

        # Save checkpoint
        save_path = os.path.join(
            args.save_dir,
            f"prober_ep-{epoch+1:02d}_lr-{args.lr}_acc-{val_metrics['acc']:.4f}_fpr-{val_metrics['fpr']:.4f}.pth"
        )
        torch.save(prober.state_dict(), save_path)

        # Early stopping
        if val_metrics['fpr'] > 90 and epoch > 30:
            print(f"Early stopping at epoch {epoch+1}, FPR: {val_metrics['fpr']:.4f}")
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
    parser.add_argument("--train-ratio", type=float, default=None, required=False)
    parser.add_argument("--label-smoothing", type=float, default=0.2)
    parser.add_argument("--loss-weight", type=float, default=4)
    parser.add_argument("--latent-dims", type=int, nargs='+', default=[256, 128, 64])
    parser.add_argument("--split", type=str, default="mirror")
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    train_set = get_dataset('hidden', hidden_data_path=args.train_path)
    hidden_mean, hidden_std = train_set.mean, train_set.std
    val_set = get_dataset('hidden', hidden_data_path=args.valid_path, mean=hidden_mean, std=hidden_std)
    

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
        hidden_dims=args.latent_dims
    ).to(args.device)
    print(prober)


    train(args, train_loader, val_loader, prober, prober_train_set, prober_val_set)


if __name__ == "__main__":
    main()
