
import argparse
import json
import os
import pickle
import sys
import random
import numpy as np
from torchmetrics.classification import BinaryStatScores

import signal
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from pytorch_lightning import seed_everything
from torch import nn
from torchvision.utils import make_grid
from tqdm.auto import tqdm
from utils.get_classifier import evaluate, get_classifier_MNIST
from utils.get_data import get_correct_labels, get_dataset, get_index_tn
from utils.get_generator import CombinedNN, Identity, adv_attack, get_generator_MNIST
from utils.get_prober import ProberMNIST
from run_orig.utils.get_generator import evaluate_all
from torchvision.utils import save_image

import wandb
from counterfactuals.data import get_data_info
from counterfactuals.plot import plot_grid_part


plt.rcParams.update(plt.rcParamsDefault)

def signal_handler(sig, frame):
    signal.signal(sig, signal.SIG_IGN)
    sys.exit(0)


def signal_handler(sig, frame):
    signal.signal(sig, signal.SIG_IGN)
    sys.exit(0)


def main(args):

    run = wandb.init(project=f"counterfactual_{args.dataset}", config=args)
    seed = 0
    random.seed(seed)
    torch.manual_seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    with open(args.index_path, "r") as f:
        split_indices = json.load(f)

    for key, val in split_indices.items():
        print(key, len(val), val[:5], val[-5:])

    prober_valid_indices = np.array([i - 60000 for i in split_indices["prober_valid"]])

    if args.dataset == "MNIST":
        classifier, data_info = get_classifier_MNIST(args.cls_ckpt_path)
        prober = ProberMNIST()
        prober.load_state_dict(torch.load(args.prober_ckpt_path))
        #print(prober)

        classifier2, data_info = get_classifier_MNIST(args.cls_ckpt_path)
        classifier2.fc_layer[5] = Identity()
        classifier2.fc_layer[6] = Identity()

        g_model = get_generator_MNIST(args.g_ckpt_path, device=args.device)

        hidden_mean, hidden_std = 0.94484913, 1.8451711

        valid_dataset = torchvision.datasets.MNIST(
            root=args.data_dir, train=False, transform=torchvision.transforms.ToTensor()
        )
        valid_dataset.data = valid_dataset.data[prober_valid_indices]
        valid_dataset.targets = valid_dataset.targets[prober_valid_indices]
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=args.batch_size, shuffle=False
        )

        train_dataset = torchvision.datasets.MNIST(
            root=args.data_dir, train=True, transform=torchvision.transforms.ToTensor()
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=False
        )

    net = CombinedNN(
        classifier2, prober, hidden_mean=hidden_mean, hidden_std=hidden_std
    )
    net.eval()

    classifier.to(args.device)
    net.to(args.device)
    classifier.eval()
    net.eval()

    train_vals = evaluate_all(train_loader, classifier, net, args.device)
    valid_vals = evaluate_all(valid_loader, classifier, net, args.device)

    metric = BinaryStatScores()
    stats = metric(valid_vals["prober_pred"], valid_vals["correct"])
    print("Verify with wandb log !!!")
    print(f'⭐️⭐️⭐️⭐️⭐️⭐️⭐️ {stats} ⭐️⭐️⭐️⭐️⭐️⭐️⭐️')

    
    ###### True Miss Case
    interested_indices = torch.where(
        (train_vals["label"] != train_vals["cls_pred"])
        & (train_vals["prober_pred"] == 0)
    )[0]
    print(f'True Neg: {len(interested_indices)}')
    
    ##### False Miss Case
    # interested_indices = torch.where(
    #     (valid_vals["label"] == valid_vals["cls_pred"])
    #     & (valid_vals["prober_pred"] == 0)
    # )[0]
    # interested_indices = torch.where((valid_vals["prober_pred"] == 0))[0]
    # print(f'False Neg: {len(interested_indices)}')



    wandb.config.update(args, allow_val_change=True)
    print(wandb.config)

    total_list = []
    
    for idx in tqdm(interested_indices):
        
        sample_info = {}
        
        print(idx)

        img_org = valid_vals["image"][idx]
        label = valid_vals["label"][idx]

        result_images, titles, cmaps, _, step_diff = adv_attack(
            img_org.clone().unsqueeze(0),
            label,
            g_model,
            classifier,
            args.device,
            data_info,
            num_steps=5000,
            lr=1e-2,
            save_at=0.9,
            target_class=label.item(),
            maximize=False,
            attack_style="z",
            attack_type="diff",
        )

        rs, ts, cm, x_prime, step_prober = adv_attack(
            img_org.clone().unsqueeze(0),
            label,
            g_model,
            net,
            args.device,
            data_info,
            num_steps=5000,
            lr=1e-2,
            save_at=0.92,
            target_class=1,
            maximize=False,
            attack_style="z",
            attack_type="prober",
        )
        result_images.extend(rs)
        titles.extend(ts)
        cmaps.extend(cm)

        prober_changed_class = torch.argmax(classifier(x_prime)).detach().cpu()
        titles[6] = f"{titles[6]} ({titles[1]} -> {prober_changed_class})"    
        
        sample_info['orig_img'] = img_org
        sample_info['label'] = label.item()
        sample_info['r_img'] = torch.Tensor(rs[2]).unsqueeze(dim=0)
        sample_info['orig_max_prob'] = valid_vals["cls_prob"][idx].max()
        total_list.append(sample_info)
    

        
        
        fig, axes = plt.subplots(2, len(result_images) // 2, figsize=(20, 10))
        axes = axes.flatten()
        for i, (img, title, cmap) in enumerate(zip(result_images, titles, cmaps)):
            axes[i].imshow(img, cmap=cmap)
            axes[i].set_title(title)
        
        fig.suptitle(
            f"Split : Valid - Cls. X, Prober X ({step_diff}/{step_prober})"
        )
        fig.savefig(os.path.join(args.save_dir, f"idx-{idx}.png"))
        plt.close(fig)
        plt.close('all')
        print("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
    with open(f"{args.save_dir}/r_data.pickle","wb") as fw:
        pickle.dump(total_list, fw)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="./<data path>")
    parser.add_argument("--dataset", type=str, default="MNIST")
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./outputs/<save path>",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument(
        "--cls-ckpt-path",
        type=str,
        default="./outputs/<classifier path>",
    )
    parser.add_argument(
        "--prober-ckpt-path",
        type=str,
        default="./outputs/<prober path>",
    )
    parser.add_argument(
        "--g-ckpt-path",
        type=str,
        default="./outputs/<generator path>",
    )
    parser.add_argument(
        "--index-path",
        type=str,
        default="./utils/split.json",
    )

    args = parser.parse_args()

    main(args)
