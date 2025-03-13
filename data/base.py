from torch.utils.data import Dataset, DataLoader
from typing import Dict, Any, Tuple

class BaseDataModule:
    """Base class for all data modules"""
    def __init__(self, data_dir: str, batch_size: int = 32):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_dataset = None
        self.val_dataset = None
        self.info = self._get_data_info()
        
    def _get_data_info(self) -> Dict[str, Any]:
        """Get dataset information"""
        raise NotImplementedError
        
    def setup(self):
        """Set up the datasets"""
        raise NotImplementedError
        
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
        )
        
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
        ) 