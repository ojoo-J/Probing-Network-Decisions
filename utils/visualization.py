import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from typing import List, Optional, Union

plt.rcParams["savefig.bbox"] = "tight"

def get_inverse_normalize(mean, std):
    """Get inverse normalization transform"""
    return torchvision.transforms.Normalize(
        mean=[-m / s for m, s in zip(mean, std)], 
        std=[1 / s for s in std]
    )

def show_images(
    imgs: Union[torch.Tensor, List[torch.Tensor]], 
    figsize: tuple = (10, 10),
    inv_normalize: Optional[torchvision.transforms.Normalize] = None
):
    """Show images in a grid"""
    if not isinstance(imgs, list):
        imgs = [imgs]
        
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False, figsize=figsize)
    
    for i, img in enumerate(imgs):
        if isinstance(img, torch.Tensor):
            img = img.detach()
        elif isinstance(img, np.ndarray):
            img = torch.from_numpy(img)
            
        if img.dim() == 2:
            img = img.unsqueeze(0)
            
        if inv_normalize is not None:
            img = inv_normalize(img)
            
        img = torchvision.transforms.functional.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        
    return fig, axs 