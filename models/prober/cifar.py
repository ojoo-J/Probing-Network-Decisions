from ..base import BaseModel
import torch.nn as nn

class ProberCIFAR10(BaseModel):
    def __init__(self, num_hidden=8192):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(num_hidden, 2048),
            nn.ReLU(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 2),
        )

    def forward(self, x):
        x = x.float()
        return self.layers(x.view(x.size(0), -1))

class ProberCIFAR100(BaseModel):
    def __init__(self, num_hidden=512):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(num_hidden, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        x = x.float()
        return self.layers(x.view(x.size(0), -1)) 