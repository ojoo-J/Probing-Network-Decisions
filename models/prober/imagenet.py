from ..base import BaseModel
import torch.nn as nn

class ProberImageNette(BaseModel):
    def __init__(self, num_hidden=2048, out_hidden=2048):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(num_hidden, out_hidden),
            nn.ReLU(),
            nn.Linear(out_hidden, 512),
            nn.ReLU(),
            nn.Linear(512, 2),
        )

    def forward(self, x):
        x = x.float()
        return self.layers(x.view(x.size(0), -1)) 