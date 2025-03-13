from ..base import BaseModel
import torch.nn as nn

class ProberMNIST(BaseModel):
    def __init__(self, latent_dim=[256, 128, 64]):
        super().__init__()
        ls = []
        for i in range(len(latent_dim)-1):
            ls.append(nn.Linear(latent_dim[i], latent_dim[i+1]))
            ls.append(nn.ReLU())
        ls.append(nn.Linear(latent_dim[-1], 2))
        self.layers = nn.Sequential(*ls)

    def forward(self, x):
        x = x.float()
        return self.layers(x.view(x.size(0), -1))

class ProberFashionMNIST(ProberMNIST):
    """Same architecture as MNIST prober"""
    pass 