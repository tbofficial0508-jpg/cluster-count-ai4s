from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _resize_like(source: Tensor, reference: Tensor) -> Tensor:
    if source.shape[-2:] == reference.shape[-2:]:
        return source
    return F.interpolate(source, size=reference.shape[-2:], mode="bilinear", align_corners=False)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class DensityCountingCNN(nn.Module):
    def __init__(self, base_channels: int = 16, dropout: float = 0.10) -> None:
        super().__init__()
        self.enc1 = ConvBlock(1, base_channels, dropout=0.0)
        self.enc2 = ConvBlock(base_channels, base_channels * 2, dropout=dropout)
        self.bottleneck = ConvBlock(base_channels * 2, base_channels * 4, dropout=dropout)
        self.dec2 = ConvBlock(base_channels * 4 + base_channels * 2, base_channels * 2, dropout=dropout)
        self.dec1 = ConvBlock(base_channels * 2 + base_channels, base_channels, dropout=dropout)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.head = nn.Conv2d(base_channels, 1, kernel_size=1)
        nn.init.zeros_(self.head.weight)
        nn.init.constant_(self.head.bias, -7.0)

    def forward(self, x: Tensor) -> Tensor:
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        bottleneck = self.bottleneck(self.pool(enc2))

        up2 = _resize_like(F.interpolate(bottleneck, scale_factor=2, mode="bilinear", align_corners=False), enc2)
        dec2 = self.dec2(torch.cat([up2, enc2], dim=1))

        up1 = _resize_like(F.interpolate(dec2, scale_factor=2, mode="bilinear", align_corners=False), enc1)
        dec1 = self.dec1(torch.cat([up1, enc1], dim=1))
        return F.softplus(self.head(dec1))


def density_to_count(density_map: Tensor) -> Tensor:
    return density_map.sum(dim=(-2, -1))


def build_model_from_checkpoint(checkpoint: dict) -> DensityCountingCNN:
    model_kwargs = checkpoint.get("model_kwargs", {})
    model = DensityCountingCNN(**model_kwargs)
    model.load_state_dict(checkpoint["state_dict"])
    return model


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> tuple[DensityCountingCNN, dict]:
    checkpoint = torch.load(Path(path), map_location=device)
    model = build_model_from_checkpoint(checkpoint)
    model.to(device)
    return model, checkpoint
