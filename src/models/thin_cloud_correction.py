from typing import Union, Tuple, Optional
import torch
import torch.nn as nn


from src.models.spatial_attention import SpatialAttentionGenerator


class ThinCloudCorrection(nn.Module):
    """Thin Cloud and Haze Correction Module with optional Spatial Attention (SpA-GAN) backbone."""

    def __init__(
        self,
        in_channels: int = 3,
        use_spatial_attention: bool = False,
        num_blocks: int = 4,
    ):
        super().__init__()
        self.use_spatial_attention = use_spatial_attention

        if use_spatial_attention:
            self.spa_gen = SpatialAttentionGenerator(
                in_channels=in_channels,
                density_channels=1,
                out_channels=in_channels,
                num_blocks=num_blocks,
            )
            self.correction = None
            self.gate = None
        else:
            self.spa_gen = None
            self.correction = nn.Sequential(
                nn.Conv2d(in_channels + 1, 32, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, in_channels, 3, padding=1),
            )
            self.gate = nn.Sequential(
                nn.Conv2d(in_channels + 1, 1, 3, padding=1),
                nn.Sigmoid(),
            )

    def forward(
        self,
        cloudy: torch.Tensor,
        density: torch.Tensor,
        return_attn: bool = False,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if self.use_spatial_attention and self.spa_gen is not None:
            return self.spa_gen(cloudy, density, return_attn=return_attn)

        x = torch.cat([cloudy, density], dim=1)
        delta = self.correction(x)
        gate = self.gate(x)
        corrected = torch.clamp(cloudy + gate * delta, 0.0, 1.0)
        if return_attn:
            return corrected, gate
        return corrected


class CloudCorrectionPipeline(nn.Module):
    def __init__(self, density_model: nn.Module, correction_model: nn.Module,
                 thin_threshold: float = 0.5):
        super().__init__()
        self.density_model = density_model
        self.correction_model = correction_model
        self.thin_threshold = thin_threshold

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        density = self.density_model(x)
        corrected = x.clone()
        thin_mask = density < self.thin_threshold
        if thin_mask.any():
            corrected = self.correction_model(x, density)
        return corrected, density
