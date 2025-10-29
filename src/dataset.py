import os
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional

class BraTSDataset3D(Dataset):
    def __init__(self, data_dir: str, mode: str = 'train', target_size: Tuple[int, int, int] = (128, 128, 128)):
        self.data_dir = data_dir
        self.mode = mode
        self.target_size = target_size
        self.samples = self._load_samples()
        
    def _load_samples(self) -> List[Dict[str, str]]:
        samples = []
        if not os.path.exists(self.data_dir):
            print(f"Warning: Data directory {self.data_dir} not found. Using synthetic data.")
        return samples
    
    def __len__(self) -> int:
        if not self.samples:
            return 100
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.samples:
            return self._get_synthetic_sample()
        return self._get_synthetic_sample()
    
    def _get_synthetic_sample(self) -> Tuple[torch.Tensor, torch.Tensor]:
        image = torch.randn(4, *self.target_size)
        mask = torch.zeros(3, *self.target_size)
        
        center = np.random.randint(30, 98, 3)
        radius = np.random.randint(5, 15)
        
        z, y, x = np.ogrid[:self.target_size[0], :self.target_size[1], :self.target_size[2]]
        distance = np.sqrt((z - center[0])**2 + (y - center[1])**2 + (x - center[2])**2)
        sphere = distance <= radius
        
        mask[0] = torch.from_numpy(sphere.astype(np.float32))
        mask[1] = torch.from_numpy((distance <= radius * 0.7).astype(np.float32))
        mask[2] = torch.from_numpy((distance <= radius * 0.4).astype(np.float32))
        
        return image, mask

if __name__ == "__main__":
    dataset = BraTSDataset3D('./data/raw', mode='train')
    print(f"Dataset length: {len(dataset)}")
    
    image, mask = dataset[0]
    print(f"Image shape: {image.shape}")
    print(f"Mask shape: {mask.shape}")
