import torch
from torch.utils.data import DataLoader
from src.dataset import BraTSDataset3D
import numpy as np
import random

def setup_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def create_data_loaders(data_dir: str, batch_size: int = 8, target_size: Tuple[int, int, int] = (128, 128, 128)):
    setup_seed(42)
    
    full_dataset = BraTSDataset3D(data_dir=data_dir, mode='train', target_size=target_size)
    
    dataset_size = len(full_dataset)
    val_size = int(0.2 * dataset_size)
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    return train_loader, val_loader

if __name__ == "__main__":
    train_loader, val_loader = create_data_loaders('./data/raw', batch_size=2)
    
    for batch_idx, (images, masks) in enumerate(train_loader):
        print(f"Batch {batch_idx}: Images: {images.shape}, Masks: {masks.shape}")
        if batch_idx >= 2:
            break
