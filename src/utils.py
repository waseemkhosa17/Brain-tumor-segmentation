import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import json

def calculate_metrics(predictions: torch.Tensor, targets: torch.Tensor, return_all: bool = False):
    if predictions.dim() == targets.dim() + 1:
        probabilities = F.softmax(predictions, dim=1)
        pred_classes = torch.argmax(probabilities, dim=1)
    else:
        pred_classes = predictions
    
    if targets.dim() == pred_classes.dim() + 1:
        target_classes = torch.argmax(targets, dim=1)
    else:
        target_classes = targets
    
    pred_flat = pred_classes.contiguous().view(-1).cpu().numpy()
    target_flat = target_classes.contiguous().view(-1).cpu().numpy()
    
    dice_scores = []
    iou_scores = []
    
    for class_idx in range(1, int(pred_flat.max()) + 1):
        pred_binary = (pred_flat == class_idx).astype(np.float32)
        target_binary = (target_flat == class_idx).astype(np.float32)
        
        if target_binary.sum() == 0 and pred_binary.sum() == 0:
            dice_scores.append(1.0)
            iou_scores.append(1.0)
            continue
        
        intersection = (pred_binary * target_binary).sum()
        dice = (2. * intersection) / (pred_binary.sum() + target_binary.sum() + 1e-8)
        dice_scores.append(dice)
        
        union = (pred_binary | target_binary).sum()
        iou = intersection / (union + 1e-8)
        iou_scores.append(iou)
    
    avg_dice = np.mean(dice_scores) if dice_scores else 0.0
    avg_iou = np.mean(iou_scores) if iou_scores else 0.0
    
    return avg_dice, avg_iou

def save_checkpoint(state, is_best: bool, filename: str = 'checkpoint.pth'):
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(os.path.dirname(filename), 'best_model.pth')
        torch.save(state, best_filename)
        print(f"Saved best model to {best_filename}")

def load_checkpoint(filename: str, model: torch.nn.Module, optimizer = None):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Checkpoint not found: {filename}")
    
    checkpoint = torch.load(filename, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Loaded checkpoint from {filename}")
    return checkpoint

def plot_training_history(history: dict, save_path: str = None):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Val Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(history['train_dice'], label='Train Dice')
    axes[0, 1].plot(history['val_dice'], label='Val Dice')
    axes[0, 1].set_title('Dice Score')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    axes[1, 0].plot(history['train_iou'], label='Train IoU')
    axes[1, 0].plot(history['val_iou'], label='Val IoU')
    axes[1, 0].set_title('IoU')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    axes[1, 1].plot(history['learning_rate'], label='Learning Rate')
    axes[1, 1].set_title('Learning Rate')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved training plots to {save_path}")
    
    plt.show()

if __name__ == "__main__":
    pred = torch.randn(2, 3, 32, 32, 32)
    target = torch.randint(0, 3, (2, 32, 32, 32))
    
    dice, iou = calculate_metrics(pred, target)
    print(f"Dice: {dice:.4f}, IoU: {iou:.4f}")
