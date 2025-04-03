import argparse
import os
import pickle
import random
import sys
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from tqdm import tqdm

# Add project root to system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import get_classifier
from data.datasets import get_dataset
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
    parser.add_argument("--layer-name", type=str, default="fc1")
    parser.add_argument("--acc-n", type=int, default=5)
    parser.add_argument("--normalize", action="store_true")
    return parser.parse_args()

def setup_model(args):
    """Setup model and hooks for feature extraction"""
    model = get_classifier(args.dataset, ckpt_path=args.ckpt_path)
    model.eval()
    model.cuda()

    features = {}
    def get_features(name):
        def hook(model, input, output):
            features[name] = output.detach()
        return hook

    # Helper function to get layer from model
    def get_layer(model, layer_name):
        # Handle dot notation for nested layers (e.g., 'conv_layer.0')
        if '.' in layer_name:
            parent_name, child_name = layer_name.split('.', 1)
            parent = getattr(model, parent_name)
            if isinstance(parent, nn.Sequential):
                return parent[int(child_name)]
            return get_layer(parent, child_name)
        return getattr(model, layer_name)

    try:
        # Try to get layer directly or through Sequential
        layer = get_layer(model, args.layer_name)
        layer.register_forward_hook(get_features(args.layer_name))
    except (AttributeError, IndexError):
        raise ValueError(f"Could not find layer: {args.layer_name}")

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

    # Get dataset using new interface
    dataset = get_dataset(
        name=args.dataset,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        normalize=args.normalize
    )
    train_loader, val_loader = dataset.get_loaders()

    # Setup model
    model, features = setup_model(args)

    # Evaluate
    evaluate(args, model, features, train_loader, "train")
    evaluate(args, model, features, val_loader, "valid")

if __name__ == "__main__":
    main()
