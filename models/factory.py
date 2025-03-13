import torch
from .cifar import CIFAR10CNN, CIFARResNet
from .mnist import MNISTCNN
from .imagenet import ImageNetResNet
from .generator import MNISTFlow
from .combined import CombinedNN, Identity
from .prober import (
    ProberMNIST, ProberFashionMNIST,
    ProberCIFAR10, ProberCIFAR100,
    ProberImageNette
)

def get_classifier(dataset: str, **kwargs):
    """Factory function to get classifier models"""
    if dataset == "MNIST":
        model = MNISTCNN()
        if kwargs.get("ckpt_path"):
            model.load_pretrained(kwargs["ckpt_path"])
            
    elif dataset == "FashionMNIST":
        model = MNISTCNN(in_channels=1, num_classes=10)
        if kwargs.get("ckpt_path"):
            model.load_pretrained(kwargs["ckpt_path"])
            
    elif dataset == "CIFAR10":
        if kwargs.get("arch") == "cnn":
            model = CIFAR10CNN(num_classes=10)
        else:
            model = CIFARResNet(
                arch=kwargs.get("arch", "resnet18"), 
                num_classes=10
            )
        if kwargs.get("pretrained", True):
            model.load_pretrained(kwargs.get("ckpt_path"))
            
    elif dataset == "CIFAR100":
        model = CIFARResNet(
            arch=kwargs.get("arch", "resnet18"), 
            num_classes=100
        )
        if kwargs.get("pretrained", True):
            model.load_pretrained(kwargs.get("ckpt_path"))
            
    elif dataset == "ImageNette":
        model = ImageNetResNet(
            arch=kwargs.get("arch", "resnet18"),
            num_classes=10,
            input_size=kwargs.get("input_size", 128)
        )
        if kwargs.get("pretrained", True):
            model.load_pretrained(kwargs.get("ckpt_path"))
            
    elif dataset == "ImageNet":
        model = ImageNetResNet(
            arch=kwargs.get("arch", "resnet18"),
            num_classes=1000,
            input_size=kwargs.get("input_size", 224)
        )
        if kwargs.get("pretrained", True):
            model.load_pretrained(kwargs.get("ckpt_path"))
            
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
        
    return model


# For backward compatibility
def get_classifier_MNIST(ckpt_path=None):
    return get_classifier("MNIST", ckpt_path=ckpt_path)

def get_classifier_FashionMNIST(ckpt_path=None):
    return get_classifier("FashionMNIST", ckpt_path=ckpt_path)

def get_classifier_CIFAR10(ckpt_path=None, arch="resnet18", pretrained=True):
    return get_classifier("CIFAR10", ckpt_path=ckpt_path, arch=arch, pretrained=pretrained)

def get_classifier_CIFAR100(ckpt_path=None, arch="resnet18", pretrained=True):
    return get_classifier("CIFAR100", ckpt_path=ckpt_path, arch=arch, pretrained=pretrained)

def get_classifier_IMAGENETTE(ckpt_path=None, arch="resnet18", pretrained=True, input_size=128):
    return get_classifier("ImageNette", ckpt_path=ckpt_path, arch=arch, pretrained=pretrained, input_size=input_size)

def get_classifier_IMAGENET(ckpt_path=None, arch="resnet18", pretrained=True, input_size=224):
    return get_classifier("ImageNet", ckpt_path=ckpt_path, arch=arch, pretrained=pretrained, input_size=input_size)

def get_generator(dataset: str, **kwargs):
    """Factory function to get generator models"""
    if dataset == "MNIST":
        model = MNISTFlow(device=kwargs.get("device", "cuda"))
        if kwargs.get("ckpt_path"):
            model.load_pretrained(kwargs["ckpt_path"])
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    return model

# For backward compatibility
def get_generator_MNIST(ckpt_path, device="cuda"):
    return get_generator("MNIST", ckpt_path=ckpt_path, device=device)

def get_prober(dataset: str, **kwargs):
    """Factory function to get prober models"""
    if dataset == "MNIST":
        model = ProberMNIST(**kwargs)
    elif dataset == "FashionMNIST":
        model = ProberFashionMNIST(**kwargs)
    elif dataset == "CIFAR10":
        model = ProberCIFAR10(**kwargs)
    elif dataset == "CIFAR100":
        model = ProberCIFAR100(**kwargs)
    elif dataset == "ImageNette":
        model = ProberImageNette(**kwargs)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
        
    if kwargs.get("ckpt_path"):
        model.load_pretrained(kwargs["ckpt_path"])
    return model 