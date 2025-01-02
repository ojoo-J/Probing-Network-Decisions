import timm
import torch
import torch.nn as nn
from utils.get_data import get_data_info


def rename_attribute(obj, old_name, new_name):
    obj._modules[new_name] = obj._modules.pop(old_name)


class CIFAR10_CNN(nn.Module):
    """
    CNN for ten class MNIST classification
    """

    def __init__(self):
        super(CIFAR10_CNN, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Conv Layer block 2
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),
        )

        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(4096, 512),
            nn.ReLU(inplace=False),
            nn.Linear(512, 256),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.1),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        out = self.conv_layer(x)
        out = out.view(out.size(0), -1)
        out = self.fc_layer(out)
        return out


def get_classifier_MNIST(ckpt_path=None):
    import sys

    sys.path.append("/project/counterfactuals")

    import counterfactuals.classifiers.cnn as classifiers

    data_info = get_data_info(dataset="MNIST", normalize=True)
    c_type = "MNIST_CNN"
    classifier = getattr(classifiers, c_type)()
    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path)
        classifier.load_state_dict(ckpt["state_dict"])
    print(classifier)
    return classifier, data_info


def get_classifier_FashionMNIST(ckpt_path=None):
    import sys

    sys.path.append("/project/counterfactuals")

    import counterfactuals.classifiers.cnn as classifiers

    data_info = get_data_info(dataset="FashionMNIST", normalize=True)
    c_type = "MNIST_CNN"
    classifier = getattr(classifiers, c_type)()
    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path)
        classifier.load_state_dict(ckpt["state_dict"])
    print(classifier)
    return classifier, data_info


def get_classifier_IMAGENETTE(ckpt_path=None, arch="xresnet50", pretrained=True):
    from fastai.layers import MaxPool
    from fastai.vision.all import xse_resnext50
    from torch.nn.modules.activation import Mish

    data_info = get_data_info(dataset="ImageNette", normalize=True)
    if arch == "xresnet50":
        classifier = xse_resnext50(n_out=10, act_cls=Mish, sa=1, sym=0, pool=MaxPool)
        if (pretrained) and (ckpt_path is not None):
            classifier.load_state_dict(torch.load(ckpt_path))
            print(f"Loaded {arch} from {ckpt_path}!")
    print(classifier)
    return classifier, data_info


def get_classifier_IMAGENET(ckpt_path=None, arch="resnet50", pretrained=True):
    data_info = get_data_info(dataset="ImageNet", normalize=True)
    if arch == "resnet18":
        cfg = {
            "num_classes": data_info["num_classes"],
            "pretrained": False,
            "input_size": (3, 160, 160),
            "pool_size": (4, 4),
            "crop_pct": 1,
            "interpolation": "bilinear",
            "first_conv": "conv1",
            "classifier": "fc",
        }
        classifier = timm.create_model(arch, pretrained=False)
        classifier.default_cfg = cfg
        classifier.pretrained_cfg = cfg

        feature_dim = 512
        classifier.fc = nn.Linear(feature_dim, data_info["num_classes"])
        if pretrained:
            if ckpt_path is None:
                classifier.load_state_dict(
                    torch.hub.load_state_dict_from_url(
                        "https://huggingface.co/nateraw/timm-resnet18-imagenette-160px-5-epochs/resolve/main/pytorch_model.bin",
                        map_location="cpu",
                        file_name="resnet18-imagenette-160px-5-epochs.pth",
                    )
                )
                print(
                    "Loaded resnet18-imagenette-160px-5-epochs from huggingface hub !"
                )
            else:
                classifier.load_state_dict(torch.load(ckpt_path)["state_dict"])
                print(f"Loaded {arch} from {ckpt_path}!")


def get_classifier_CIFAR100(ckpt_path=None, arch="resnet18", pretrained=True):
    data_info = get_data_info(dataset="CIFAR100", normalize=True)
    if arch == "vgg16":
        cfg = {
            "num_classes": 100,
            "pretrained": False,
            "input_size": (3, 32, 32),
            "pool_size": (4, 4),
            "crop_pct": 1,
            "interpolation": "bilinear",
            "first_conv": "features.0",
            "classifier": "head.fc",
        }
        classifier = timm.create_model("vgg16_bn", pretrained=False)
        classifier.default_cfg = cfg
        classifier.pretrained_cfg = cfg

        classifier.pre_logits = nn.Identity()  # type: ignore
        features_dim = 512
        classifier.head.fc = nn.Linear(features_dim, data_info["num_classes"])

        if pretrained:
            if ckpt_path is None:
                classifier.load_state_dict(
                    torch.hub.load_state_dict_from_url(
                        "https://huggingface.co/edadaltocg/vgg16_bn_cifar100/resolve/main/pytorch_model.bin",
                        map_location="cpu",
                        progress=True,
                        file_name="vgg16_bn_cifar100.pth",
                    )
                )
                print("Loaded vgg16_bn_cifar100 from huggingface hub !")
            else:
                ckpt = torch.load(ckpt_path)
                if "state_dict" in ckpt:
                    classifier.load_state_dict(torch.load(ckpt_path)["state_dict"])
                else:
                    # classifier.pre_logits = nn.Flatten()
                    # rename_attribute(classifier, "head", "classifier")
                    # classifier.classifier = nn.Sequential(
                    #     nn.Linear(512, 512),
                    #     nn.ReLU(True),
                    #     nn.Dropout(),
                    #     nn.Linear(512, 512),
                    #     nn.ReLU(True),
                    #     nn.Dropout(),
                    #     nn.Linear(512, 100),  # data_info["num_classes"]),
                    # )
                    # classifier.load_state_dict(torch.load(ckpt_path))
                    # rename_attribute(classifier, "classifier", "head")

                    classifier.load_state_dict(torch.load(ckpt_path))
                print(f"Loaded {arch} from {ckpt_path}!")

    elif arch == "resnet18":
        cfg = {
            "num_classes": data_info["num_classes"],
            "pretrained": False,
            "input_size": (3, 32, 32),
            "pool_size": (4, 4),
            "crop_pct": 1,
            "interpolation": "bilinear",
            "first_conv": "conv1",
            "classifier": "fc",
        }
        classifier = timm.create_model(arch, pretrained=False)
        classifier.default_cfg = cfg
        classifier.pretrained_cfg = cfg
        classifier.conv1 = nn.Conv2d(
            3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
        )
        classifier.maxpool = nn.Identity()  # type: ignore
        feature_dim = 512
        classifier.fc = nn.Linear(feature_dim, data_info["num_classes"])
        if pretrained:
            if ckpt_path is None:
                classifier.load_state_dict(
                    torch.hub.load_state_dict_from_url(
                        "https://huggingface.co/edadaltocg/resnet18_cifar100/resolve/main/pytorch_model.bin",
                        map_location="cpu",
                        file_name="resnet18_cifar100.pth",
                    )
                )
                print("Loaded resnet18_cifar100 from huggingface hub !")
            else:
                classifier.load_state_dict(torch.load(ckpt_path)["state_dict"])
                print(f"Loaded {arch} from {ckpt_path}!")

    elif arch == "resnet34":
        cfg = {
            "num_classes": data_info["num_classes"],
            "pretrained": False,
            "input_size": (3, 32, 32),
            "pool_size": (4, 4),
            "crop_pct": 1,
            "interpolation": "bilinear",
            "first_conv": "conv1",
            "classifier": "fc",
        }
        classifier = timm.create_model(arch, pretrained=False)
        classifier.default_cfg = cfg
        classifier.pretrained_cfg = cfg
        classifier.conv1 = nn.Conv2d(
            3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
        )
        classifier.maxpool = nn.Identity()  # type: ignore
        feature_dim = 512
        classifier.fc = nn.Linear(feature_dim, data_info["num_classes"])
        if pretrained:
            if ckpt_path is None:
                classifier.load_state_dict(
                    torch.hub.load_state_dict_from_url(
                        "https://huggingface.co/edadaltocg/resnet34_cifar100/resolve/main/pytorch_model.bin",
                        map_location="cpu",
                        file_name="resnet34_cifar100.pth",
                    )
                )
                print("Loaded resnet34_cifar100 from huggingface hub !")
            else:
                classifier.load_state_dict(torch.load(ckpt_path)["state_dict"])
                print(f"Loaded {arch} from {ckpt_path}!")

    elif arch == "resnet50":
        cfg = {
            "num_classes": data_info["num_classes"],
            "pretrained": False,
            "input_size": (3, 32, 32),
            "pool_size": (4, 4),
            "crop_pct": 1,
            "interpolation": "bilinear",
            "first_conv": "conv1",
            "classifier": "fc",
        }
        classifier = timm.create_model(arch, pretrained=False)
        classifier.default_cfg = cfg
        classifier.pretrained_cfg = cfg
        classifier.conv1 = nn.Conv2d(
            3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
        )
        classifier.maxpool = nn.Identity()  # type: ignore
        feature_dim = 2048
        classifier.fc = nn.Linear(feature_dim, data_info["num_classes"])
        if pretrained:
            if ckpt_path is None:
                classifier.load_state_dict(
                    torch.hub.load_state_dict_from_url(
                        "https://huggingface.co/edadaltocg/resnet50_cifar100/resolve/main/pytorch_model.bin",
                        map_location="cpu",
                        file_name="resnet50_cifar100.pth",
                    )
                )
                print("Loaded resnet50_cifar100 from huggingface hub !")
            else:
                classifier.load_state_dict(torch.load(ckpt_path)["state_dict"])
                print(f"Loaded {arch} from {ckpt_path}!")

    print(classifier)
    return classifier, data_info


def get_classifier_CIFAR10(ckpt_path=None, arch="resnet18", pretrained=True):
    data_info = get_data_info(dataset="CIFAR10", normalize=True)

    if arch == "cnn":
        classifier = CIFAR10_CNN()
        if pretrained and (ckpt_path is not None):
            classifier.load_state_dict(torch.load(ckpt_path)["state_dict"])
            print(f"Loaded {arch} from {ckpt_path}!")

    elif arch == "resnet18":
        cfg = {
            "num_classes": 10,
            "pretrained": False,
            "input_size": (3, 32, 32),
            "pool_size": (4, 4),
            "crop_pct": 1,
            "interpolation": "bilinear",
            "first_conv": "conv1",
            "classifier": "fc",
        }
        classifier = timm.create_model(arch, pretrained=False)
        classifier.default_cfg = cfg
        classifier.pretrained_cfg = cfg
        classifier.conv1 = nn.Conv2d(
            3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
        )
        classifier.maxpool = nn.Identity()  # type: ignore
        classifier.fc = nn.Linear(512, data_info["num_classes"])
        if pretrained:
            if ckpt_path is None:
                classifier.load_state_dict(
                    torch.hub.load_state_dict_from_url(
                        "https://huggingface.co/edadaltocg/resnet18_cifar10/resolve/main/pytorch_model.bin",
                        map_location="cuda:1",
                        file_name="resnet18_cifar10.pth",
                    )
                )
                print("Loaded resnet18_cifar10 from huggingface hub !")
            else:
                classifier.load_state_dict(torch.load(ckpt_path)["state_dict"])
                print(f"Loaded {arch} from {ckpt_path}!")
    elif arch == "resnet34":
        cfg = {
            "num_classes": 10,
            "pretrained": False,
            "input_size": (3, 32, 32),
            "pool_size": (4, 4),
            "crop_pct": 1,
            "interpolation": "bilinear",
            "first_conv": "conv1",
            "classifier": "fc",
        }
        classifier = timm.create_model(arch, pretrained=False)
        classifier.default_cfg = cfg
        classifier.pretrained_cfg = cfg
        classifier.conv1 = nn.Conv2d(
            3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
        )
        classifier.maxpool = nn.Identity()  # type: ignore
        classifier.fc = nn.Linear(512, data_info["num_classes"])
        if pretrained:
            if ckpt_path is None:
                classifier.load_state_dict(
                    torch.hub.load_state_dict_from_url(
                        "https://huggingface.co/edadaltocg/resnet34_cifar10/resolve/main/pytorch_model.bin",
                        map_location="cpu",
                        file_name="resnet34_cifar10.pth",
                    )
                )
                print("Loaded resnet34_cifar10 from huggingface hub !")
            else:
                classifier.load_state_dict(torch.load(ckpt_path)["state_dict"])
                print(f"Loaded {arch} from {ckpt_path}!")
    elif arch == "vgg16":
        cfg = {
            "num_classes": 10,
            "pretrained": False,
            "input_size": (3, 32, 32),
            "pool_size": (4, 4),
            "crop_pct": 1,
            "interpolation": "bilinear",
            "first_conv": "features.0",
            "classifier": "head.fc",
        }
        classifier = timm.create_model("vgg16_bn", pretrained=False)
        classifier.default_cfg = cfg
        classifier.pretrained_cfg = cfg
        features_dim = 512
        classifier.pre_logits = nn.Identity()  # type: ignore
        classifier.head.fc = nn.Linear(features_dim, data_info["num_classes"])
        if pretrained:
            if ckpt_path is None:
                classifier.load_state_dict(
                    torch.hub.load_state_dict_from_url(
                        "https://huggingface.co/edadaltocg/vgg16_bn_cifar10/resolve/main/pytorch_model.bin",
                        map_location="cpu",
                        progress=True,
                        file_name="vgg16_bn_cifar10.pth",
                    )
                )
                print("Loaded vgg16_bn_cifar10 from huggingface hub !")
            else:
                classifier.load_state_dict(torch.load(ckpt_path)["state_dict"])
                print(f"Loaded {arch} from {ckpt_path}!")
    print(classifier)
    return classifier, data_info


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()

    images = []
    labels = []
    preds = []

    for x, y in dataloader:
        images.append(x)
        labels.append(y)

        pred = model(x.to(device))
        _, pred_class = pred.max(dim=1)

        preds.append(pred_class.detach().cpu())

    images = torch.cat(images).detach().cpu()
    labels = torch.cat(labels).detach().cpu()
    preds = torch.cat(preds)

    acc = (labels == preds).sum() / len(preds)
    print(f"Acc : {acc:.4f}")
    return images, labels, preds, acc
