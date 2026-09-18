import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mpconv(x)


class Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class MCDropout(nn.Module):
    def __init__(self, p: float = 0.1):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.dropout2d(x, self.p, training=True)


class CloudDensityNet(nn.Module):
    """Cloud Density estimation network with Early Topographic DEM Fusion.

    Features:
    - 4-level U-Net backbone with Monte Carlo Dropout for uncertainty estimation.
    - Early topographic DEM fusion at the second contraction level (down1 feature map)
      to differentiate terrain shadow boundaries from genuine cloud shadows.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        features: list[int] = None,
        dropout_p: float = 0.0,
        dem_channels: int = 0,
    ):
        super().__init__()
        if features is None:
            features = [32, 64, 128, 256]
        self.features = features
        self.dem_channels = dem_channels

        self.inc = DoubleConv(in_channels, features[0])
        self.down1 = Down(features[0], features[1])

        # Early fusion projection at the 2nd contraction level if configured
        if dem_channels > 0:
            self.dem_fusion = nn.Sequential(
                nn.Conv2d(features[1] + dem_channels, features[1], kernel_size=1),
                nn.BatchNorm2d(features[1]),
                nn.ReLU(inplace=True),
            )
        else:
            self.dem_fusion = None

        self.down2 = Down(features[1], features[2])
        self.down3 = Down(features[2], features[3])
        self.down4 = Down(features[3], features[3])

        self.up1 = Up(features[3] + features[3], features[2])
        self.up2 = Up(features[2] + features[2], features[1])
        self.up3 = Up(features[1] + features[1], features[0])
        self.up4 = Up(features[0] + features[0], features[0])

        self.outc = nn.Conv2d(features[0], out_channels, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = MCDropout(dropout_p) if dropout_p > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, dem: torch.Tensor = None) -> torch.Tensor:
        """Forward pass with optional DEM early fusion.

        Args:
            x: Input optical imagery tensor (B, C, H, W).
            dem: Optional topographic DEM feature tensor (B, dem_channels, H, W).
                 If provided, early-fused at the second contraction level.
        """
        x1 = self.inc(x)
        x2 = self.down1(x1)  # 2nd contraction level

        # Early Topographic DEM Fusion
        if dem is not None:
            if dem.shape[1] == 1:
                from src.preprocessing.dem_features import compute_dem_feature_stack
                dem_feats = compute_dem_feature_stack(dem)
            else:
                dem_feats = dem

            if self.dem_channels > 0 and dem_feats.shape[1] > self.dem_channels:
                dem_feats = dem_feats[:, :self.dem_channels]

            # Match spatial dimensions of x2 (H/2, W/2)
            if dem_feats.shape[2:] != x2.shape[2:]:
                dem_down = F.interpolate(dem_feats, size=(x2.size(2), x2.size(3)), mode="bilinear", align_corners=False)
            else:
                dem_down = dem_feats

            if self.dem_fusion is not None and dem_down.shape[1] == self.dem_channels:
                x2 = self.dem_fusion(torch.cat([x2, dem_down], dim=1))
            else:
                x2 = x2 + dem_down[:, :1].expand(-1, x2.size(1), -1, -1) * 0.05

        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        x = self.outc(x)
        x = self.dropout(x)
        return self.sigmoid(x)

    @torch.no_grad()
    def predict_with_uncertainty(
        self, x: torch.Tensor, n_samples: int = 20, dem: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        preds = torch.stack([self(x, dem=dem) for _ in range(n_samples)], dim=0)
        mean = preds.mean(dim=0)
        std = preds.std(dim=0)
        return mean, std
