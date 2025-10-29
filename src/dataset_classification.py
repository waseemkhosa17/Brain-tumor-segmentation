import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from typing import List, Tuple
import glob

class BrainTumorClassificationDataset(Dataset):
    """Dataset for Brain Tumor Classification from JPEG images"""
    
    def __init__(self, data_dir: str, transform=None, mode: str = 'train'):
        self.data_dir = data_dir
        self.transform = transform
        self.mode = mode
        self.classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.samples = self._load_samples()
    
    def _load_samples(self) -> List[Tuple[str, int]]:
        """Load all image paths and their labels"""
        samples = []
        
        if self.mode == 'train':
            base_path = os.path.join(self.data_dir, 'Training')
        else:
            base_path = os.path.join(self.data_dir, 'Testing')
        
        for class_name in self.classes:
            class_dir = os.path.join(base_path, class_name)
            if not os.path.exists(class_dir):
                print(f"Warning: Directory {class_dir} not found")
                continue
                
            # Get all jpg files
            image_files = glob.glob(os.path.join(class_dir, '*.jpg'))
            for img_path in image_files:
                samples.append((img_path, self.class_to_idx[class_name]))
        
        print(f"Loaded {len(samples)} samples for {self.mode}")
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_transforms(mode: str = 'train'):
    """Get data transformations for training and validation"""
    from torchvision import transforms
    
    if mode == 'train':
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])

if __name__ == "__main__":
    # Test the dataset
    dataset = BrainTumorClassificationDataset('./data/raw', mode='train')
    print(f"Dataset classes: {dataset.classes}")
    print(f"Number of samples: {len(dataset)}")
    
    if len(dataset) > 0:
        image, label = dataset[0]
        print(f"Image shape: {image.shape}")
        print(f"Label: {label} - {dataset.classes[label]}")