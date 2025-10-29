import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import json
from tqdm import tqdm
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from src.model_nnunet import nnUNet3D
from src.data_loader import create_data_loaders
from src.losses import get_loss_function
from src.utils import save_checkpoint, load_checkpoint, calculate_metrics, plot_training_history

class Trainer:
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = nnUNet3D(
            in_channels=config.get('in_channels', 4),
            out_channels=config.get('out_channels', 3),
            base_channels=config.get('base_channels', 32),
            depth=config.get('depth', 5)
        )
        self.model.to(self.device)
        
        self.criterion = get_loss_function(config.get('loss_fn', 'combined'))
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config.get('lr', 1e-4))
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=10)
        
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_dice': [], 'val_dice': [],
            'train_iou': [], 'val_iou': [],
            'learning_rate': []
        }
        
        os.makedirs('models', exist_ok=True)
        os.makedirs('outputs/visualizations', exist_ok=True)
        
        print(f"Training on device: {self.device}")
    
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss, total_dice, total_iou, num_batches = 0.0, 0.0, 0.0, 0
        
        pbar = tqdm(train_loader, desc='Training')
        for images, masks in pbar:
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            loss.backward()
            self.optimizer.step()
            
            dice, iou = calculate_metrics(outputs, masks)
            
            total_loss += loss.item()
            total_dice += dice
            total_iou += iou
            num_batches += 1
            
            pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Dice': f'{dice:.4f}'})
        
        return total_loss / num_batches, total_dice / num_batches, total_iou / num_batches
    
    def validate_epoch(self, val_loader):
        self.model.eval()
        total_loss, total_dice, total_iou, num_batches = 0.0, 0.0, 0.0, 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc='Validation')
            for images, masks in pbar:
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                dice, iou = calculate_metrics(outputs, masks)
                
                total_loss += loss.item()
                total_dice += dice
                total_iou += iou
                num_batches += 1
                
                pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Dice': f'{dice:.4f}'})
        
        return total_loss / num_batches, total_dice / num_batches, total_iou / num_batches
    
    def train(self, train_loader, val_loader):
        start_epoch = 0
        best_dice = 0.0
        
        checkpoint_path = os.path.join('models', 'checkpoint.pth')
        if os.path.exists(checkpoint_path):
            checkpoint = load_checkpoint(checkpoint_path, self.model, self.optimizer)
            start_epoch = checkpoint['epoch']
            best_dice = checkpoint['best_dice']
            self.history = checkpoint['history']
            print(f"Resumed training from epoch {start_epoch}")
        
        print("Starting training...")
        
        for epoch in range(start_epoch, self.config['epochs']):
            print(f"\nEpoch {epoch + 1}/{self.config['epochs']}")
            print("-" * 50)
            
            train_loss, train_dice, train_iou = self.train_epoch(train_loader)
            val_loss, val_dice, val_iou = self.validate_epoch(val_loader)
            
            current_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(val_loss)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_dice'].append(train_dice)
            self.history['val_dice'].append(val_dice)
            self.history['train_iou'].append(train_iou)
            self.history['val_iou'].append(val_iou)
            self.history['learning_rate'].append(current_lr)
            
            print(f"Train Loss: {train_loss:.4f}, Dice: {train_dice:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Dice: {val_dice:.4f}")
            print(f"Learning Rate: {current_lr:.2e}")
            
            is_best = val_dice > best_dice
            if is_best:
                best_dice = val_dice
                print(f"New best Dice: {best_dice:.4f}")
            
            save_checkpoint({
                'epoch': epoch + 1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best_dice': best_dice,
                'history': self.history,
                'config': self.config
            }, is_best, filename=checkpoint_path)
            
            with open('outputs/training_history.json', 'w') as f:
                json.dump(self.history, f, indent=2)
            
            if (epoch + 1) % 10 == 0:
                plot_training_history(self.history, 'outputs/training_plots.png')
        
        print("Training completed!")
        plot_training_history(self.history, 'outputs/training_plots.png')

def main():
    config = {
        'in_channels': 4,
        'out_channels': 3,
        'base_channels': 32,
        'depth': 5,
        'lr': 1e-4,
        'epochs': 100,
        'batch_size': 8,
        'loss_fn': 'combined',
        'target_size': (128, 128, 128)
    }
    
    train_loader, val_loader = create_data_loaders('./data/raw', batch_size=config['batch_size'])
    trainer = Trainer(config)
    trainer.train(train_loader, val_loader)

if __name__ == "__main__":
    main()
