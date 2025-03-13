import torch
import torch.nn as nn
from typing import Optional, Dict, Any

class BaseModel(nn.Module):
    """Base class for all models"""
    def __init__(self):
        super().__init__()
        self.default_cfg: Optional[Dict[str, Any]] = None
        
    def load_pretrained(self, path):
        """Load pretrained weights"""
        if path is None:
            return
            
        # Load checkpoint
        checkpoint = torch.load(path)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
            
        self.load_state_dict(state_dict)

    def load_pretrained_from_hub(self, ckpt_path: Optional[str] = None, map_location: str = "cpu") -> None:
        """Load pretrained weights from huggingface hub"""
        if ckpt_path is None:
            if not hasattr(self, 'hub_url'):
                raise ValueError("No checkpoint path provided and no default hub URL defined")
            self.load_state_dict(
                torch.hub.load_state_dict_from_url(
                    self.hub_url,
                    map_location=map_location,
                    file_name=self.hub_filename
                )
            )
            print(f"Loaded {self.__class__.__name__} from huggingface hub!")
        else:
            ckpt = torch.load(ckpt_path, map_location=map_location)
            state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
            self.load_state_dict(state_dict)
            print(f"Loaded {self.__class__.__name__} from {ckpt_path}!") 