import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    """Two conv layers with Instance Norm and Leaky ReLU"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )
    def forward(self, x):
        return self.block(x)


class EncoderBlock(nn.Module):
    """ConvBlock + MaxPool downsampling"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        skip = self.conv(x)   # save for skip connection
        down = self.pool(skip)
        return skip, down


class DecoderBlock(nn.Module):
    """Upsample + concatenate skip + ConvBlock"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up   = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_ch * 2, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)  # concatenate skip connection
        return self.conv(x)


class NNUNet3D(nn.Module):
    """
    3D nnU-Net Architecture
    Input : (B, 4, 128, 128, 128)  - 4 MRI modalities
    Output: (B, 4, 128, 128, 128)  - 4 classes (BG, NCR, ED, ET)
    """
    def __init__(self, in_channels=4, out_channels=4, base_ch=32):
        super().__init__()
        f = base_ch  # 32

        # Encoder (5 levels)
        self.enc1 = EncoderBlock(in_channels, f)      # 32
        self.enc2 = EncoderBlock(f,           f*2)    # 64
        self.enc3 = EncoderBlock(f*2,         f*4)    # 128
        self.enc4 = EncoderBlock(f*4,         f*8)    # 256

        # Bottleneck
        self.bottleneck = ConvBlock(f*8, f*16)        # 512

        # Decoder (4 levels)
        self.dec4 = DecoderBlock(f*16, f*8)           # 256
        self.dec3 = DecoderBlock(f*8,  f*4)           # 128
        self.dec2 = DecoderBlock(f*4,  f*2)           # 64
        self.dec1 = DecoderBlock(f*2,  f)             # 32

        # Final output layer
        self.final = nn.Conv3d(f, out_channels, kernel_size=1)

        # Weight initialization
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="leaky_relu")
            elif isinstance(m, nn.InstanceNorm3d) and m.affine:
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # Encoder path
        s1, x = self.enc1(x)
        s2, x = self.enc2(x)
        s3, x = self.enc3(x)
        s4, x = self.enc4(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder path with skip connections
        x = self.dec4(x, s4)
        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)

        return self.final(x)


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters    : {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    return trainable
