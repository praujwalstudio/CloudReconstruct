"""Spatial Attention Generator (SpA-GAN) for Thin Cloud & Haze Removal.

Inspired by Spatial Attention GAN (SpA-GAN):
Uses Spatial Attention Blocks (SAB) to generate continuous soft attention maps
A in [0, 1], directing the network's capacity strictly onto cloud-affected areas
while leaving clear land surface pixels completely preserved.
"""

from typing import Tuple, Union, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAttentionBlock(nn.Module):
    """Spatial Attention Block (SAB).

    Learns a spatial attention mask A in [0, 1] to selectively gate
    and enhance features in cloud/haze-affected regions.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
        )
        self.attn_conv = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        residual = x
        feat = self.conv1(x)
        feat = self.conv2(feat)
        attn = self.attn_conv(feat)
        out = residual + feat * attn
        return self.relu(out), attn


class SpatialAttentionGenerator(nn.Module):
    """Spatial Attention Residual Generator (SpA-GAN style) for thin cloud removal."""

    def __init__(
        self,
        in_channels: int = 3,
        density_channels: int = 1,
        out_channels: int = 3,
        num_features: int = 64,
        num_blocks: int = 4,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.density_channels = density_channels
        self.out_channels = out_channels

        total_in = in_channels + density_channels

        # Shallow feature extraction
        self.in_conv = nn.Sequential(
            nn.Conv2d(total_in, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Deep Spatial Attention Residual backbone
        self.blocks = nn.ModuleList([
            SpatialAttentionBlock(num_features) for _ in range(num_blocks)
        ])

        # Feature fusion and reconstruction head
        self.fusion = nn.Sequential(
            nn.Conv2d(num_features * (num_blocks + 1), num_features, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features),
            nn.ReLU(inplace=True),
        )

        self.out_delta = nn.Conv2d(num_features, out_channels, kernel_size=3, padding=1)
        self.final_attn = nn.Sequential(
            nn.Conv2d(num_features, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        cloudy: torch.Tensor,
        density: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Reconstruct clear optical surface from cloudy input and optional density map.

        Args:
            cloudy: Cloudy optical image (B, C, H, W)
            density: Optional cloud density/probability map (B, 1, H, W)
            return_attn: If True, returns (reconstructed, attention_map)

        Returns:
            Reconstructed clear image (B, C, H, W) [and attention map (B, 1, H, W)]
        """
        if density is None:
            density = torch.zeros(
                cloudy.shape[0], self.density_channels, cloudy.shape[2], cloudy.shape[3],
                device=cloudy.device, dtype=cloudy.dtype
            )

        inp = torch.cat([cloudy, density], dim=1)
        x = self.in_conv(inp)

        feats = [x]
        attns = []
        for block in self.blocks:
            x, attn = block(x)
            feats.append(x)
            attns.append(attn)

        fused = self.fusion(torch.cat(feats, dim=1))
        delta = self.out_delta(fused)
        overall_attn = self.final_attn(fused)

        # Apply spatial attention gating: clean = cloudy + A * delta
        corrected = cloudy + overall_attn * delta
        corrected = torch.clamp(corrected, 0.0, 1.0)

        if return_attn:
            return corrected, overall_attn
        return corrected
