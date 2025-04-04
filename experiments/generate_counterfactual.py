import argparse
import json
import os
import pickle
import random
import numpy as np
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryStatScores

import signal
import sys

# Add project root to system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from tqdm.auto import tqdm
from models import (
    get_classifier,
    get_prober,
    get_generator,
    CombinedNN,
)
from data import get_dataset, split_dataset
from counterfactuals.utils import (
    expl_to_image,
    torch_to_image,
)
from utils.metrics import calculate_metrics

def run_adv_attack(x,
                   z,
                   optimizer,
                   classifier,
                   g_model,
                   target_class: int,
                   attack_style: str,
                   save_at: float,
                   num_steps: int,
                   maximize: bool):
    """
    run optimization process on x or z for num_steps iterations
    early stopping when save_at is reached
    if not return None
    """
    target = torch.LongTensor([target_class]).to(x.device)

    softmax = torch.nn.Softmax(dim=1)
    loss_fn = torch.nn.CrossEntropyLoss()

    with tqdm(total=num_steps) as progress_bar:
        for step in range(num_steps):
            optimizer.zero_grad()

            if attack_style == "z":
                x = g_model.decode(z)

            # assert that x is a valid image
            x.data = torch.clip(x.data, min=0.0, max=1.0)

            if "UNet" in type(classifier).__name__:
                _, regression = classifier(x)
                # minimize negative regression to maximize regression
                loss = -regression if maximize else regression

                progress_bar.set_postfix(regression=regression.item(), loss=loss.item(), step=step + 1)
                progress_bar.update()

                if (maximize and regression.item() > save_at) or (not maximize and regression.item() < save_at):
                    return x

            else:
                prediction = classifier(x)
                acc = softmax(prediction)[torch.arange(0, x.shape[0]), target]
                loss = loss_fn(prediction, target)

                progress_bar.set_postfix(acc_target=acc.item(), loss=loss.item(), step=step + 1)
                progress_bar.update()

                # early stopping
                if acc > save_at:
                    return x

            loss.backward()
            optimizer.step()

    return x


def adv_attack(
        img_org,
        label,
        generator,
        model,
        device,
        data_shape,
        num_steps=5000,
        lr=5e-5,
        save_at=0.99,
        target_class=None,
        maximize=False,
        attack_style="z",
        attack_type="diff"
    ):
    img_org = img_org.to(device)
    x = img_org.clone().to(device)

    # define parameters that will be optimized
    params = []
    if attack_style == "z":
        # define z as params for derivative wrt to z
        z = generator.encode(x)
        z = [z_i.detach() for z_i in z] if isinstance(z, list) else z.detach()
        z_org = [z_i.clone() for z_i in z] if isinstance(z, list) else z.clone()

        if type(z) == list:
            for z_part in z:
                z_part.requires_grad = True
                params.append(z_part)
        else:
            z.requires_grad = True
            params.append(z)
    else:
        # define x as params for derivative wrt x
        x.requires_grad = True
        params.append(x)
        z = None

    print(
        "\nRunning counterfactual search in Z ..."
        if attack_style == "z"
        else "Running conventional adv attack in X ..."
    )
    optimizer = torch.optim.Adam(params=params, lr=lr, weight_decay=0.0)

    # run the adversarial attack
    x_prime = run_adv_attack(
        x,
        z,
        optimizer,
        model,
        generator,
        target_class,
        attack_style,
        save_at,
        num_steps,
        maximize,
    )

    if x_prime is None:
        print(
            "Warning: Maximum number of iterations exceeded! "
            "Attack did not reach target value, returned None."
        )

    # save results
    class_names = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    cmap_img = "jet" if data_shape[0] == 3 else "gray"

    # calculate heatmap as difference dx between original and adversarial/counterfactual
    heatmap = torch.abs(img_org - x_prime).sum(dim=0).sum(dim=0)
    all_images = [torch_to_image(img_org)]

    pred_class = class_names[torch.argmax(model(img_org)).detach().cpu()]
    adv_class = class_names[torch.argmax(model(x_prime)).detach().cpu()]

    titles = [f"({label}){pred_class}", adv_class, "$\delta x$"]
    cmaps = [cmap_img, cmap_img, "coolwarm"]
    if attack_style == "z":
        recov_x = generator.decode(z_org)
        all_images.append(torch_to_image(recov_x))
        recov_class = class_names[torch.argmax(model(recov_x)).detach().cpu()]
        titles = [f"({label}){pred_class}", recov_class, adv_class, "$\delta x$"]
        cmaps = [cmap_img, cmap_img, cmap_img, "coolwarm"]

    all_images.append(torch_to_image(x_prime))
    all_images.append(expl_to_image(heatmap))

    return all_images, titles, cmaps, x_prime

def evaluate_all(loader, classifier, net, device, image_shape=(1, 28, 28), num_classes=10):
    """Evaluate models on data loader"""
    vals = {}
    # Initialize lists to store results
    images_list = []
    labels_list = []
    clf_out_list = []
    clf_prob_list = []
    clf_pred_list = []
    prb_out_list = []
    prb_prob_list = []
    prb_pred_list = []
    correct_list = []

    for batch in loader:
        # Handle different batch formats
        if len(batch) == 5:  # (idx, hidden, correct, img, label)
            _, _, _, img, target = batch
        elif len(batch) == 3:  # (idx, img, label)
            _, img, target = batch
        else:  # (img, label)
            img, target = batch
            
        img = img.to(device)
        target = target.to(device)

        # Get classifier predictions
        cls_out = classifier(img)
        cls_prob = torch.softmax(cls_out, dim=1)
        cls_pred = torch.argmax(cls_prob, dim=1)
        
        # Get prober predictions
        prb_out = net(img)
        prb_prob = torch.softmax(prb_out, dim=1)
        prb_pred = torch.argmax(prb_prob, dim=1)
        
        # Store results in lists
        images_list.append(img)
        labels_list.append(target)
        clf_out_list.append(cls_out)
        clf_prob_list.append(cls_prob)
        clf_pred_list.append(cls_pred)
        prb_out_list.append(prb_out)
        prb_prob_list.append(prb_prob)
        prb_pred_list.append(prb_pred)
        correct_list.append((cls_pred == target))

    # Concatenate all results into tensors
    vals["image"] = torch.cat([img.detach().cpu() for img in images_list])
    vals["label"] = torch.cat([target.detach().cpu() for target in labels_list])
    vals["clf_out"] = torch.cat([out.detach().cpu() for out in clf_out_list])
    vals["clf_prob"] = torch.cat([prob.detach().cpu() for prob in clf_prob_list])
    vals["clf_pred"] = torch.cat([pred.detach().cpu() for pred in clf_pred_list])
    vals["prb_out"] = torch.cat([out.detach().cpu() for out in prb_out_list])
    vals["prb_prob"] = torch.cat([prob.detach().cpu() for prob in prb_prob_list])
    vals["prb_pred"] = torch.cat([pred.detach().cpu() for pred in prb_pred_list])
    vals["correct"] = torch.cat([correct.detach().cpu() for correct in correct_list])

    return vals

def main(args):
    # Set random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    # Load split indices
    with open(args.index_path, "r") as f:
        split_indices = json.load(f)
    prober_valid_indices = np.array([i - 60000 for i in split_indices["prober_valid"]])

    # Get dataset and models
    dataset = get_dataset(
        name=args.dataset,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        normalize=args.normalize
    )
    data_shape = dataset.val_dataset[0][0].shape
    print(f"data shape : {data_shape}")

    # Get models
    classifier = get_classifier(args.dataset, ckpt_path=args.cls_ckpt_path).to(args.device)
    prober = get_prober(
        args.dataset,
        hidden_dims=args.prober_dims,
        ckpt_path=args.prober_ckpt_path
    ).to(args.device)
    generator = get_generator(args.dataset, ckpt_path=args.g_ckpt_path).to(args.device)

    # Get data loaders
    train_loader, val_loader = dataset.get_loaders()

    # Create validation dataset with specific indices
    train_set = get_dataset('hidden', hidden_data_path=args.prober_train_path)
    hidden_mean, hidden_std = train_set.mean, train_set.std
    val_set = get_dataset('hidden', hidden_data_path=args.prober_valid_path, mean=hidden_mean, std=hidden_std)
    
    # Only support for mirror
    # train_set, val_set = dataset.train_dataset, dataset.val_dataset
    # train_set, val_set, split_dict = split_dataset(
    #     train_set, val_set, args.prober_train_ratio, None, args.prober_split
    # )

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True)

    # Create combined model
    net = CombinedNN(
        classifier=classifier,
        prober=prober,
        hidden_mean=hidden_mean,
        hidden_std=hidden_std,
        prober_layer_name=args.prober_layer_name,
    )

    # Move models to device
    classifier.to(args.device).eval()
    net.to(args.device).eval()
    
    # # Evaluate models
    train_vals = evaluate_all(train_loader, classifier, net, args.device)
    valid_vals = evaluate_all(val_loader, classifier, net, args.device)

    train_metrics = calculate_metrics(train_loader, net, in_type='image', out_type='correct')
    val_metrics = calculate_metrics(val_loader, net, in_type='image', out_type='correct')
    print(f"[Train] Acc: {train_metrics['acc']:.4f}, F1: {train_metrics['f1']:.4f}, FPR: {train_metrics['fpr']:.4f}")
    print(f"[Val] Acc: {val_metrics['acc']:.4f}, F1: {val_metrics['f1']:.4f}, FPR: {val_metrics['fpr']:.4f}")

    # Calculate metrics
    metric = BinaryStatScores()
    stats = metric(valid_vals["prb_pred"], valid_vals["correct"])
    print(f'⭐️ Stats: {stats} ⭐️')

    # Find interesting cases and generate counterfactuals
    interested_indices = torch.where(
        (valid_vals["label"] != valid_vals["clf_pred"]) &
        (valid_vals["prb_pred"] == 0)
    )[0]
    interested_indices = sorted(interested_indices)
    print(f'True Negatives: {len(interested_indices)}')
    
        ##### False Miss Case
    interested_indices = torch.where(
        (valid_vals["label"] == valid_vals["clf_pred"])
        & (valid_vals["prb_pred"] == 0)
    )[0]
    interested_indices = torch.where((valid_vals["prb_pred"] == 0))[0]
    interested_indices = sorted(interested_indices)
    print(f'False Neg: {len(interested_indices)}')

    # # Generate and save counterfactuals
    os.makedirs(args.save_dir, exist_ok=True)
    total_list = []

    for idx in tqdm(interested_indices):
        img_org = valid_vals["image"][idx]
        label = valid_vals["label"][idx]

        # Generate counterfactuals 
        # Conventional CF
        result_images, titles, cmaps, _ = adv_attack(
            img_org.clone().unsqueeze(0).to(args.device),
            label,
            generator,
            classifier,
            args.device,
            data_shape,
            num_steps=args.steps,
            lr=1e-2,
            save_at=0.9,
            target_class=label.item(),
            maximize=False,
            attack_style="z",
            attack_type="diff",
        )

        # Implicit CF
        rs, ts, cm, x_prime = adv_attack(
            img_org.clone().unsqueeze(0),
            label,
            generator,
            net,
            args.device,
            data_shape,
            num_steps=args.steps,
            lr=args.lr,
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
        
        # Save results
        sample_info = {
            'orig_img': img_org,
            'label': label.item(),
            'r_img': torch.Tensor(rs[2]).unsqueeze(dim=0),
            'orig_max_prob': valid_vals["clf_prob"][idx].max()
        }
        total_list.append(sample_info)
    
        # Visualize results
        fig, axes = plt.subplots(2, len(result_images) // 2, figsize=(20, 10))
        axes = axes.flatten()
        for i, (img, title, cmap) in enumerate(zip(result_images, titles, cmaps)):
            axes[i].imshow(img, cmap=cmap)
            axes[i].set_title(title)
        
        fig.suptitle(
            f"Idx: {idx}, Split : Valid - Cls. X, Prober X )"
        )
        fig.savefig(os.path.join(args.save_dir, f"idx-{idx}.png"))
        plt.close(fig)
        plt.close('all')

    # Save all results
    with open(os.path.join(args.save_dir, "r_data.pickle"), "wb") as fw:
        pickle.dump(total_list, fw)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="./<data path>")
    parser.add_argument("--dataset", type=str, default="MNIST")
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./outputs/<save path>",
    )
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
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
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--prober-dims", type=int, nargs='+', default=[256, 128, 64])
    parser.add_argument(
        "--prober-train-path",
        type=str,
        default="/project/run/outputs/<classifier path for train dataset>",
    )
    parser.add_argument(
        "--prober-valid-path",
        type=str,
        default="/project/outputs/<classifier path for val dataset>",
    )
    parser.add_argument("--prober-split", type=str, default="mirror")
    parser.add_argument("--prober-train-ratio", type=float, default=None, required=False)
    parser.add_argument("--prober-layer-name", type=str, default="fc1")


    args = parser.parse_args()

    main(args)
