import torch
import torch.nn as nn
import torchvision.models as models

class BrainTumorClassifier(nn.Module):
    """CNN Model for Brain Tumor Classification"""
    
    def __init__(self, num_classes: int = 4, pretrained: bool = True):
        super(BrainTumorClassifier, self).__init__()
        
        # Use pre-trained ResNet18
        self.backbone = models.resnet18(pretrained=pretrained)
        
        # Replace the final fully connected layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

class SimpleCNN(nn.Module):
    """Simple CNN for Brain Tumor Classification"""
    
    def __init__(self, num_classes: int = 4):
        super(SimpleCNN, self).__init__()
        
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Fourth conv block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def get_model(model_name: str = 'resnet18', num_classes: int = 4, pretrained: bool = True):
    """Factory function to get model"""
    if model_name == 'resnet18':
        return BrainTumorClassifier(num_classes, pretrained)
    elif model_name == 'simple_cnn':
        return SimpleCNN(num_classes)
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model
    elif model_name == 'efficientnet':
        model = models.efficientnet_b0(pretrained=pretrained)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Linear(in_features, num_classes)
        return model
    else:
        raise ValueError(f"Unknown model: {model_name}")

if __name__ == "__main__":
    # Test the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = get_model('resnet18', num_classes=4)
    model.to(device)
    
    # Test input
    x = torch.randn(2, 3, 224, 224).to(device)
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")