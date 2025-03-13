import torch
from ..base import BaseModel
from counterfactuals.generative_models.factory import get_generative_model

class MNISTFlow(BaseModel):
    """Flow-based generator for MNIST"""
    def __init__(self, device="cuda"):
        super().__init__()
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
        
        self.model, self.model_type = get_generative_model(
            generative_model_type="Flow", 
            data_info=data_info, 
            device=device
        )
        
    def forward(self, x):
        return self.model(x)
        
    def encode(self, x):
        return self.model.encode(x)
        
    def decode(self, z):
        return self.model.decode(z) 