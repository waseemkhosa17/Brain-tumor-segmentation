import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import zoom

class BraTSDataset(Dataset):
    def __init__(self, data_dir, case_list, target_shape=(128,128,128), augment=False):
        self.data_dir     = data_dir
        self.cases        = case_list
        self.target_shape = target_shape
        self.augment      = augment
        self.modalities   = ["flair", "t1", "t1ce", "t2"]

    def __len__(self):
        return len(self.cases)

    def normalize(self, volume):
        """Z-score normalization on non-zero brain voxels only"""
        mask = volume > 0
        if mask.sum() == 0:
            return volume
        mean = volume[mask].mean()
        std  = volume[mask].std() + 1e-8
        volume = (volume - mean) / std
        volume[~mask] = 0
        return volume

    def resize(self, volume, order=1):
        """Resize volume to target shape"""
        factors = [t / s for t, s in zip(self.target_shape, volume.shape)]
        return zoom(volume, factors, order=order)

    def remap_labels(self, seg):
        """
        BraTS original labels: 0=BG, 1=NCR, 2=ED, 4=ET
        Remap to:              0=BG, 1=NCR, 2=ED, 3=ET
        """
        new_seg = np.zeros_like(seg, dtype=np.int64)
        new_seg[seg == 1] = 1
        new_seg[seg == 2] = 2
        new_seg[seg == 4] = 3
        return new_seg

    def augment_data(self, image, seg):
        """Random flips along each axis"""
        for axis in range(1, 4):  # axes 1,2,3 (skip channel axis 0)
            if np.random.rand() > 0.5:
                image = np.flip(image, axis=axis).copy()
                seg   = np.flip(seg,   axis=axis-1).copy()
        return image, seg

    def __getitem__(self, idx):
        case      = self.cases[idx]
        case_path = os.path.join(self.data_dir, case)

        # Load and process 4 modalities
        imgs = []
        for mod in self.modalities:
            path = os.path.join(case_path, f"{case}_{mod}.nii.gz")
            vol  = nib.load(path).get_fdata().astype(np.float32)
            vol  = self.normalize(vol)
            vol  = self.resize(vol, order=1)
            imgs.append(vol)

        image = np.stack(imgs, axis=0)  # shape: (4, 128, 128, 128)

        # Load and process segmentation mask
        seg_path = os.path.join(case_path, f"{case}_seg.nii.gz")
        seg = nib.load(seg_path).get_fdata().astype(np.float32)
        seg = self.resize(seg, order=0)  # order=0 for nearest-neighbor (labels)
        seg = self.remap_labels(seg)     # remap 4 -> 3

        # Augmentation
        if self.augment:
            image, seg = self.augment_data(image, seg)

        return (
            torch.FloatTensor(image),
            torch.LongTensor(seg)
        )


def get_loaders(data_dir, batch_size=2, val_split=0.2, num_workers=2):
    """Create train and validation DataLoaders"""
    cases = sorted([f for f in os.listdir(data_dir) if f.startswith("BraTS")])

    # Split: 80% train, 20% val
    split     = int((1 - val_split) * len(cases))
    train_cases = cases[:split]
    val_cases   = cases[split:]

    print(f"Total cases : {len(cases)}")
    print(f"Train cases : {len(train_cases)}")
    print(f"Val cases   : {len(val_cases)}")

    train_dataset = BraTSDataset(data_dir, train_cases, augment=True)
    val_dataset   = BraTSDataset(data_dir, val_cases,   augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader
