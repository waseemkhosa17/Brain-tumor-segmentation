import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if predictions.size(1) > 1:
            predictions = F.softmax(predictions, dim=1)
        
        predictions_flat = predictions.contiguous().view(-1)
        targets_flat = targets.contiguous().view(-1)
        
        intersection = (predictions_flat * targets_flat).sum()
        union = predictions_flat.sum() + targets_flat.sum()
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice

class CombinedLoss(nn.Module):
    def __init__(self, dice_weight: float = 0.7, ce_weight: float = 0.3):
        super(CombinedLoss, self).__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.dice_loss = DiceLoss()
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets_ce = torch.argmax(targets, dim=1)
        dice_loss = self.dice_loss(predictions, targets)
        ce_loss = self.ce_loss(predictions, targets_ce)
        
        return self.dice_weight * dice_loss + self.ce_weight * ce_loss

def get_loss_function(loss_name: str = 'combined', **kwargs):
    if loss_name == 'dice':
        return DiceLoss(**kwargs)
    elif loss_name == 'combined':
        return CombinedLoss(**kwargs)
    elif loss_name == 'cross_entropy':
        return nn.CrossEntropyLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")

if __name__ == "__main__":
    predictions = torch.randn(2, 3, 32, 32, 32)
    targets = torch.randint(0, 3, (2, 32, 32, 32))
    
    loss_fn = CombinedLoss()
    loss = loss_fn(predictions, targets)
    print(f"Loss: {loss.item():.4f}")
