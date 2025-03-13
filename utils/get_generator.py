import torch
from torch import nn
from collections import defaultdict
from counterfactuals.adv import run_adv_attack
from counterfactuals.generative_models.factory import get_generative_model
from counterfactuals.utils import (
    expl_to_image,
    get_transforms,
    make_dir,
    torch_to_image,
)
from utils.compute_metrics import calc_entropy

class CombinedNN(nn.Module):
    def __init__(self, classifier, prober, hidden_mean, hidden_std):
        super(CombinedNN, self).__init__()
        self.classifier = classifier
        self.prober = prober

        self.freeze(self.prober)
        self.freeze(self.classifier)
        
        self.hidden_mean = hidden_mean
        self.hidden_std = hidden_std

    def freeze(self, model):
        for param in model.parameters():
            param.requires_grad = False

    def forward(self, x):
        hidden = self.classifier(x)
        hidden = (hidden - self.hidden_mean) / self.hidden_std
        correct = self.prober(hidden)
        return correct


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def get_generator_MNIST(ckpt_path, device="cuda"):
    data_info = {
        "data_set": "MNIST",
        "data_shape": [1, 28, 28],
        "n_bits": 8,
        "temp": 1,
        "num_classes": 10,
        "class_names": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
        "data_mean": [0.1307],
        "data_std": [0.3081],
    }

    g_model, g_model_type = get_generative_model(
        generative_model_type="Flow", data_info=data_info, device=device
    )
    ckpt = torch.load(ckpt_path)
    g_model.load_state_dict(ckpt["state_dict"])
    g_model.eval()
    g_model.to(device)
    return g_model


def adv_attack(
    x,
    label,
    g_model,
    classifier,
    device,
    data_info,
    num_steps: int = 5000,
    lr: float = 5e-5,
    save_at: float = 0.99,
    target_class: int = 1,
    attack_style: str = "z",
    maximize: bool = False,
    save_fname: str = "adv_attack",
    attack_type: str = "diff",
) -> None:
    """
    prepare adversarial attack in X or Z
    run attack
    save resulting adversarial example/counterfactual
    """
    x = x.to(device)

    # define parameters that will be optimized
    params = []
    if attack_style == "z":
        # define z as params for derivative wrt to z
        z = g_model.encode(x)
        z = [z_i.detach() for z_i in z] if isinstance(z, list) else z.detach()
        x_org = x.detach().clone()
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
        x_org = x.clone()
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
        classifier,
        g_model,
        target_class,
        attack_style,
        save_at,
        num_steps,
        maximize,
    )

    if x_prime is None:
        print(
            "Warning: Maximum number of iterations exceeded! Attack did not reach target value, returned None."
        )
        x_prime = x_org

    # save results
    class_names = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    data_shape = data_info["data_shape"]
    cmap_img = "jet" if data_shape[0] == 3 else "gray"

    # calculate heatmap as difference dx between original and adversarial/counterfactual
    # TODO: dx to original or projection?
    heatmap = torch.abs(x_org - x_prime).sum(dim=0).sum(dim=0)
    all_images = [torch_to_image(x_org)]

    pred_class = class_names[torch.argmax(classifier(x_org)).detach().cpu()]
    adv_class = class_names[torch.argmax(classifier(x_prime)).detach().cpu()]

    # titles = ["$x$", "$x^\prime$", "$\delta x$"]
    titles = [f"({label}){pred_class}", adv_class, "$\delta x$"]
    cmaps = [cmap_img, cmap_img, "coolwarm"]
    if attack_style == "z":
        recov_x = g_model.decode(z_org)
        all_images.append(torch_to_image(recov_x))
        recov_class = class_names[torch.argmax(classifier(recov_x)).detach().cpu()]
        # titles = ["$x$", "$g(g^{-1}(x))$", "$x^\prime$", "$\delta x$"]
        titles = [f"({label}){pred_class}", recov_class, adv_class, "$\delta x$"]
        cmaps = [cmap_img, cmap_img, cmap_img, "coolwarm"]

    all_images.append(torch_to_image(x_prime))
    all_images.append(expl_to_image(heatmap))

    return all_images, titles, cmaps, x_prime
    # _ = plot_grid_part(all_images, titles=titles, images_per_row=4, cmap=cmaps)
    # plt.subplots_adjust(
    #     wspace=0.03, hspace=0.01, left=0.03, right=0.97, bottom=0.01, top=0.95
    # )

@torch.no_grad()
def evaluate_all(data_loader, classifier, combinedNet, device):
    import torch.nn.functional as F
    vals = defaultdict(list)
    for img, label in data_loader:
        img = img.to(device)
        label = torch.Tensor([int(l) for l in label]).to(device)

        cls_out = classifier(img)
        prober_out = combinedNet(img)

        cls_prob = F.softmax(cls_out, dim=1)
        prober_prob = F.softmax(prober_out, dim=1)

        cls_pred = torch.argmax(cls_prob, dim=1)
        prober_pred = torch.argmax(prober_prob, dim=1)

        hidden = combinedNet(img)
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
