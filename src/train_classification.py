import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import json
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from src.dataset_classification import BrainTumorClassificationDataset, get_transforms
from src.model_classification import get_model
from src.utils import save_checkpoint, load_checkpoint

class ClassificationTrainer:
    """Trainer for Brain Tumor Classification"""
    
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.model = get_model(
            model_name=config.get('model_name', 'resnet18'),
            num_classes=config.get('num_classes', 4),
            pretrained=config.get('pretrained', True)
        )
        self.model.to(self.device)
        
        # Loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.get('lr', 1e-4),
            weight_decay=config.get('weight_decay', 1e-5)
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
        
        # Training history
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'learning_rate': []
        }
        
        # Create directories
        os.makedirs('models', exist_ok=True)
        os.makedirs('outputs', exist_ok=True)
        
        print(f"Training on device: {self.device}")
        print(f"Model: {config.get('model_name', 'resnet18')}")
        print(f"Number of classes: {config.get('num_classes', 4)}")
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(train_loader, desc='Training')
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            # Calculate accuracy
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            total_loss += loss.item()
            accuracy = accuracy_score(all_labels, all_preds)
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{accuracy:.4f}'
            })
        
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = accuracy_score(all_labels, all_preds)
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self, val_loader):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc='Validation')
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                total_loss += loss.item()
                accuracy = accuracy_score(all_labels, all_preds)
                
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{accuracy:.4f}'
                })
        
        epoch_loss = total_loss / len(val_loader)
        epoch_acc = accuracy_score(all_labels, all_preds)
        
        return epoch_loss, epoch_acc, all_preds, all_labels
    
    def train(self, train_loader, val_loader):
        """Full training loop"""
        start_epoch = 0
        best_acc = 0.0
        
        # Load checkpoint if exists
        checkpoint_path = os.path.join('models', 'classification_checkpoint.pth')
        if os.path.exists(checkpoint_path):
            checkpoint = load_checkpoint(checkpoint_path, self.model, self.optimizer)
            start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            self.history = checkpoint['history']
            print(f"Resumed training from epoch {start_epoch}")
        
        print("Starting training...")
        
        for epoch in range(start_epoch, self.config['epochs']):
            print(f"\nEpoch {epoch + 1}/{self.config['epochs']}")
            print("-" * 50)
            
            # Training phase
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validation phase
            val_loss, val_acc, val_preds, val_labels = self.validate_epoch(val_loader)
            
            # Update learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(val_loss)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rate'].append(current_lr)
            
            # Print epoch summary
            print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
            print(f"Learning Rate: {current_lr:.2e}")
            
            # Save checkpoint
            is_best = val_acc > best_acc
            if is_best:
                best_acc = val_acc
                print(f"New best accuracy: {best_acc:.4f}")
                
                # Save classification report for best model
                class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
                report = classification_report(val_labels, val_preds, target_names=class_names, output_dict=True)
                with open('outputs/best_model_report.json', 'w') as f:
                    json.dump(report, f, indent=2)
            
            save_checkpoint({
                'epoch': epoch + 1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best_acc': best_acc,
                'history': self.history,
                'config': self.config
            }, is_best, filename=checkpoint_path)
            
            # Save history
            with open('outputs/training_history_classification.json', 'w') as f:
                json.dump(self.history, f, indent=2)
            
            # Plot training history every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.plot_training_history()
        
        print("Training completed!")
        self.plot_training_history()
    
    def plot_training_history(self):
        """Plot training history"""
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        ax1.plot(self.history['train_loss'], label='Train Loss')
        ax1.plot(self.history['val_loss'], label='Val Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracy
        ax2.plot(self.history['train_acc'], label='Train Accuracy')
        ax2.plot(self.history['val_acc'], label='Val Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('outputs/classification_training_plots.png', dpi=300, bbox_inches='tight')
        plt.close()

def create_data_loaders(data_dir: str, batch_size: int = 32):
    """Create data loaders for training and validation"""
    # Training dataset and loader
    train_dataset = BrainTumorClassificationDataset(
        data_dir=data_dir,
        transform=get_transforms('train'),
        mode='train'
    )
    
    # Validation dataset (using test folder)
    val_dataset = BrainTumorClassificationDataset(
        data_dir=data_dir,
        transform=get_transforms('val'),
        mode='test'
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    return train_loader, val_loader

def main():
    """Main training function"""
    config = {
        'model_name': 'resnet18',
        'num_classes': 4,
        'pretrained': True,
        'lr': 1e-4,
        'weight_decay': 1e-5,
        'epochs': 50,
        'batch_size': 32
    }
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        data_dir='./data/raw',
        batch_size=config['batch_size']
    )
    
    # Initialize trainer
    trainer = ClassificationTrainer(config)
    
    # Start training
    trainer.train(train_loader, val_loader)

if __name__ == "__main__":
    main()