"""PatchGAN Discriminator for satellite cloud removal texture evaluation.

Evaluates local N x N patches (e.g., 32x32 / 70x70) rather than full-image scalars,
enforcing sharp high-frequency textures, realistic edge contours, and spectral consistency.
"""

from typing import Optional
import torch
import torch.nn as nn


class PatchGANDiscriminator(nn.Module):
    """PatchGAN (Markovian) Discriminator for adversarial texture classification."""

    def __init__(
        self,
        in_channels: int = 3,
        condition_channels: int = 0,
        num_filters: int = 64,
        num_layers: int = 3,
    ):
        super().__init__()
        self.condition_channels = condition_channels
        total_in = in_channels + condition_channels

        layers = [
            nn.Conv2d(total_in, num_filters, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        curr_filters = num_filters
        for i in range(1, num_layers):
            next_filters = min(curr_filters * 2, 512)
            layers.extend([
                nn.Conv2d(curr_filters, next_filters, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(next_filters),
                nn.LeakyReLU(0.2, inplace=True),
            ])
            curr_filters = next_filters

        # Final classification patch layer (stride 1)
        layers.extend([
            nn.Conv2d(curr_filters, curr_filters, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(curr_filters),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(curr_filters, 1, kernel_size=3, stride=1, padding=1),
        ])

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Classify real vs fake patches.

        Args:
            x: Optical image to evaluate (B, C, H, W)
            condition: Optional condition tensor (e.g. cloudy input / SAR / density)

        Returns:
            2D grid of patch logits (B, 1, H_out, W_out)
        """
        if condition is not None:
            x = torch.cat([x, condition], dim=1)
        return self.model(x)
