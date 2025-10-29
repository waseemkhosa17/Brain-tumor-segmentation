import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

class ConvBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super(ConvBlock3D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout3d(dropout),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class EncoderBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super(EncoderBlock3D, self).__init__()
        self.conv = ConvBlock3D(in_channels, out_channels, dropout)
        self.pool = nn.MaxPool3d(2)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.conv(x)
        pooled = self.pool(x)
        return x, pooled

class DecoderBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super(DecoderBlock3D, self).__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock3D(out_channels * 2, out_channels, dropout)
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        diff = [skip.size()[i] - x.size()[i] for i in range(2, len(x.size()))]
        x = F.pad(x, [diff[2] // 2, diff[2] - diff[2] // 2,
                      diff[1] // 2, diff[1] - diff[1] // 2,
                      diff[0] // 2, diff[0] - diff[0] // 2])
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class nnUNet3D(nn.Module):
    def __init__(self, in_channels: int = 4, out_channels: int = 3, base_channels: int = 32, depth: int = 5, dropout: float = 0.3):
        super(nnUNet3D, self).__init__()
        
        self.depth = depth
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        
        for i in range(depth):
            in_ch = in_channels if i == 0 else base_channels * (2 ** (i - 1))
            out_ch = base_channels * (2 ** i)
            self.encoders.append(EncoderBlock3D(in_ch, out_ch, dropout))
        
        self.bottleneck = ConvBlock3D(base_channels * (2 ** (depth - 1)), base_channels * (2 ** depth), dropout)
        
        for i in range(depth - 1, -1, -1):
            in_ch = base_channels * (2 ** (i + 1))
            out_ch = base_channels * (2 ** i)
            self.decoders.append(DecoderBlock3D(in_ch, out_ch, dropout))
        
        self.final_conv = nn.Conv3d(base_channels, out_channels, kernel_size=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        
        for encoder in self.encoders:
            skip, x = encoder(x)
            skips.append(skip)
        
        x = self.bottleneck(x)
        
        for i, decoder in enumerate(self.decoders):
            x = decoder(x, skips[-(i + 1)])
        
        return self.final_conv(x)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nnUNet3D(in_channels=4, out_channels=3, base_channels=32, depth=5)
    model.to(device)
    
    x = torch.randn(2, 4, 128, 128, 128).to(device)
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
