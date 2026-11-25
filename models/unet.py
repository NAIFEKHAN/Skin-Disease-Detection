"""
ResNet18-Encoder U-Net Architecture for Lesion Boundary Segmentation.
Phase 3 Implementation.

Features:
- Pretrained ResNet18 feature extractor as contracting path (encoder).
- Multi-scale skip connections preserving fine lesion boundaries.
- Progressive bilateral upsampling decoder with double-convolution blocks.
- Single-channel raw logits output (loss-compatible with BCEWithLogitsLoss).
- Sigmoid activation helper for inference and probability thresholding.
- CPU and GPU execution compatibility.
"""

from typing import Optional, Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


class ConvBlock(nn.Module):
    """
    Standard double convolution block:
    [Conv2d -> BatchNorm2d -> ReLU] x 2
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DecoderBlock(nn.Module):
    """
    Progressive upsampling decoder block with skip-connection fusion:
    1. Bilinear 2x upsampling (or ConvTranspose2d).
    2. Channel concatenation with encoder skip features.
    3. Double convolution refinement.
    """
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        # Handle small rounding differences if dimensions vary slightly
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class ResNet18UNet(nn.Module):
    """
    Lightweight U-Net using a pretrained ResNet18 backbone.
    Total parameters: ~14.3 Million (lightweight and efficient for fast training).
    
    Dimensions (for input 3x256x256):
    - Skip 0 (stem):       64 channels,  128x128
    - Skip 1 (layer1):     64 channels,  64x64
    - Skip 2 (layer2):     128 channels, 32x32
    - Skip 3 (layer3):     256 channels, 16x16
    - Bottleneck (layer4): 512 channels, 8x8
    - Dec 4 (layer4+skip3):256 channels, 16x16
    - Dec 3 (dec4+skip2):  128 channels, 32x32
    - Dec 2 (dec3+skip1):  64 channels,  64x64
    - Dec 1 (dec2+skip0):  64 channels,  128x128
    - Final Up (dec1):     32 channels,  256x256
    - Output Head:         1 channel,    256x256 (raw logits)
    """
    def __init__(
        self,
        pretrained: bool = True,
        in_channels: int = 3,
        num_classes: int = 1
    ):
        super().__init__()
        # Load genuine pretrained ResNet18 backbone
        if pretrained:
            weights = ResNet18_Weights.DEFAULT
            backbone = resnet18(weights=weights)
        else:
            backbone = resnet18(weights=None)

        # Encoder stem (scale 1/2)
        if in_channels == 3:
            self.stem_conv = backbone.conv1
        else:
            self.stem_conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.stem_bn = backbone.bn1
        self.stem_relu = backbone.relu
        self.maxpool = backbone.maxpool

        # Encoder stages
        self.encoder1 = backbone.layer1  # 64 channels, 64x64
        self.encoder2 = backbone.layer2  # 128 channels, 32x32
        self.encoder3 = backbone.layer3  # 256 channels, 16x16
        self.encoder4 = backbone.layer4  # 512 channels, 8x8 (bottleneck)

        # Decoder stages with skip connections
        self.decoder4 = DecoderBlock(in_channels=512, skip_channels=256, out_channels=256)
        self.decoder3 = DecoderBlock(in_channels=256, skip_channels=128, out_channels=128)
        self.decoder2 = DecoderBlock(in_channels=128, skip_channels=64,  out_channels=64)
        self.decoder1 = DecoderBlock(in_channels=64,  skip_channels=64,  out_channels=64)

        # Final upsampling to 256x256 and segmentation head
        self.final_upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning raw logits of shape (B, 1, H, W).
        BCEWithLogitsLoss should be used with raw logits.
        """
        # Encoder (Contracting Path)
        x0 = self.stem_relu(self.stem_bn(self.stem_conv(x)))  # (B, 64, 128, 128) -> Skip 0
        x_pool = self.maxpool(x0)                             # (B, 64, 64, 64)
        x1 = self.encoder1(x_pool)                            # (B, 64, 64, 64)   -> Skip 1
        x2 = self.encoder2(x1)                                # (B, 128, 32, 32)  -> Skip 2
        x3 = self.encoder3(x2)                                # (B, 256, 16, 16)  -> Skip 3
        x4 = self.encoder4(x3)                                # (B, 512, 8, 8)    -> Bottleneck

        # Decoder (Expanding Path) with Skip Connections
        d4 = self.decoder4(x4, x3)                            # (B, 256, 16, 16)
        d3 = self.decoder3(d4, x2)                            # (B, 128, 32, 32)
        d2 = self.decoder2(d3, x1)                            # (B, 64, 64, 64)
        d1 = self.decoder1(d2, x0)                            # (B, 64, 128, 128)

        # Final head
        out = self.final_upsample(d1)                         # (B, 32, 256, 256)
        logits = self.head(out)                               # (B, 1, 256, 256)
        return logits

    def predict_mask(self, x: torch.Tensor, threshold: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inference helper: applies sigmoid and returns (probability_map, binary_mask).
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
            binary_mask = (probs >= threshold).float()
            return probs, binary_mask

    def get_parameter_count(self) -> Dict[str, int]:
        """Returns total and trainable parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total_parameters": total, "trainable_parameters": trainable}


def build_unet(
    pretrained: bool = True,
    in_channels: int = 3,
    num_classes: int = 1,
    device: Optional[torch.device] = None
) -> ResNet18UNet:
    """Factory helper to build and instantiate the ResNet18-UNet model."""
    model = ResNet18UNet(pretrained=pretrained, in_channels=in_channels, num_classes=num_classes)
    if device is not None:
        model = model.to(device)
    return model


if __name__ == "__main__":
    # Smoke test on dummy input
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_unet(pretrained=True, device=device)
    params = model.get_parameter_count()
    print(f"ResNet18-UNet initialized on {device}.")
    print(f"Total parameters: {params['total_parameters']:,}")
    print(f"Trainable parameters: {params['trainable_parameters']:,}")

    dummy_input = torch.randn(2, 3, 256, 256, device=device)
    out = model(dummy_input)
    print(f"Forward pass output shape: {out.shape} (Expected: [2, 1, 256, 256])")
    assert out.shape == (2, 1, 256, 256), "Output shape mismatch!"

    probs, binary = model.predict_mask(dummy_input, threshold=0.5)
    print(f"predict_mask output shapes: probs={probs.shape}, binary={binary.shape}")
    assert binary.shape == (2, 1, 256, 256)
    print("✔ models/unet.py architecture smoke test passed!")
