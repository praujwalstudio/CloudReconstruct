import torch
import torch.nn as nn


class ThinCloudCorrection(nn.Module):
    def __init__(self, in_channels: int = 3):
        super().__init__()
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

    def forward(self, cloudy: torch.Tensor, density: torch.Tensor) -> torch.Tensor:
        x = torch.cat([cloudy, density], dim=1)
        delta = self.correction(x)
        gate = self.gate(x)
        return cloudy + gate * delta


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
