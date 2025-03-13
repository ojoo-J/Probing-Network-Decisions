import argparse
import os
import pickle
import random
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data
from tqdm import tqdm

from models.factory import get_classifier
from data.factory import get_dataset
from utils.metrics import accuracy

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate classifier and extract hidden features")
    parser.add_argument("--dataset", type=str, default="MNIST")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--save-dir", type=str, default="outputs")
    parser.add_argument("--arch", type=str, default="resnet18")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--module-name", type=str, default="classifier")
    parser.add_argument("--layer-idx", type=int, default=4)
    parser.add_argument("--acc-n", type=int, default=5)
    return parser.parse_args()

def setup_model(args):
    """Setup model and hooks for feature extraction"""
    model = get_classifier(args.dataset, ckpt_path=args.ckpt_path, arch=args.arch)
    model.eval()
    model.cuda()

    features = {}
    def get_features(name):
        def hook(model, input, output):
            features[name] = output.detach()
        return hook

    # Register hook to get features
    if hasattr(model, args.module_name):
        module = getattr(model, args.module_name)
        if isinstance(module, nn.Sequential):
            module[args.layer_idx].register_forward_hook(
                get_features(f"{args.module_name}-{args.layer_idx}th-layer")
            )
        else:
            module.register_forward_hook(
                get_features(args.module_name)
            )
    else:
        raise ValueError(f"Model has no module named {args.module_name}")

    return model, features

def evaluate(args, model, features, data_loader, split="train"):
    """Evaluate model and save features"""
    correct = []
    hiddens = []
    images = []
    labels = []
    indices = []  # Start with empty list
    base_idx = 0  # Keep track of current index

    with torch.no_grad():
        for i, batch in enumerate(tqdm(data_loader)):
            # Handle both (img, target) and (idx, img, target) formats
            if len(batch) == 3:
                idx, img, target = batch
            else:
                img, target = batch
                # Generate sequential indices for this batch
                idx = list(range(base_idx, base_idx + len(img)))
                base_idx += len(img)

            img = img.cuda()
            target = target.cuda()

            # Forward pass
            output = model(img)
            acc = accuracy(output, target, topk=(1, args.acc_n))

            # Get features and other data
            for k, v in features.items():
                hiddens.append(v.cpu())
            correct.extend(acc[0].cpu().numpy())
            images.append(img.cpu())
            labels.extend(target.cpu().numpy())
            indices.extend(idx if isinstance(idx, list) else idx.tolist())

    # Save results
    os.makedirs(args.save_dir, exist_ok=True)
    for k in features.keys():
        data = {
            "hidden": torch.cat(hiddens),
            "correct": correct,
            "image": torch.cat(images),
            "label": labels,
            "index": indices
        }
        save_path = os.path.join(
            args.save_dir,
            f"{args.dataset}_{k}_acc{args.acc_n}_{split}_{len(indices)}.pkl"
        )
        with open(save_path, "wb") as f:
            pickle.dump(data, f)
        print(f"Saved to {save_path}")

def main():
    args = parse_args()

    # Set random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    # Get data
    train_dataset = get_dataset(
        name=args.dataset,
        root=args.data_dir,
        split='train'
    )
    val_dataset = get_dataset(
        name=args.dataset,
        root=args.data_dir,
        split='val'
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
    )

    # Setup model
    model, features = setup_model(args)

    # Evaluate
    evaluate(args, model, features, train_loader, "train")
    evaluate(args, model, features, val_loader, "valid")

if __name__ == "__main__":
    main()
