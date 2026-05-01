import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """Dice Loss for segmentation — ignores background class"""
    def __init__(self, num_classes=4, smooth=1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.smooth      = smooth

    def forward(self, pred, target):
        # pred  : (B, C, D, H, W) raw logits
        # target: (B, D, H, W)    integer labels
        pred_soft = F.softmax(pred, dim=1)

        # One-hot encode target
        target_oh = F.one_hot(target, self.num_classes)   # (B,D,H,W,C)
        target_oh = target_oh.permute(0, 4, 1, 2, 3).float()  # (B,C,D,H,W)

        # Compute dice per class (skip background class 0)
        dice_scores = []
        for c in range(1, self.num_classes):
            p = pred_soft[:, c]
            t = target_oh[:, c]
            intersection = (p * t).sum()
            union        = p.sum() + t.sum()
            dice = (2 * intersection + self.smooth) / (union + self.smooth)
            dice_scores.append(dice)

        return 1 - torch.stack(dice_scores).mean()


class CombinedLoss(nn.Module):
    """Dice Loss + Cross Entropy Loss (nnU-Net standard)"""
    def __init__(self, num_classes=4, dice_weight=0.5, ce_weight=0.5):
        super().__init__()
        self.dice_loss  = DiceLoss(num_classes)
        self.ce_loss    = nn.CrossEntropyLoss()
        self.dice_weight = dice_weight
        self.ce_weight   = ce_weight

    def forward(self, pred, target):
        dice = self.dice_loss(pred, target)
        ce   = self.ce_loss(pred, target)
        return self.dice_weight * dice + self.ce_weight * ce
