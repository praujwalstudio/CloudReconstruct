"""Deep Image Prior (DIP) Inpainting Fallback for Satellite Cloud Removal.

Inspired by training-free Deep Image Prior (DIP) formulations (e.g. Strath-AI DIP):
Uses an untrained neural network as an inductive visual prior to inpaint out-of-distribution,
heavily obscured, or low-confidence optical regions strictly using test-time optimization
over available clear/high-confidence pixels and Total Variation regularization.
"""

from typing import Optional, Union, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DIPBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DIPNet(nn.Module):
    """Lightweight untrained U-Net prior for test-time inpainting."""

    def __init__(self, in_channels: int = 8, out_channels: int = 3, hidden_dims: list = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [16, 32, 64]

        self.inc = DIPBlock(in_channels, hidden_dims[0])
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DIPBlock(hidden_dims[0], hidden_dims[1]))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DIPBlock(hidden_dims[1], hidden_dims[2]))

        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv_up1 = DIPBlock(hidden_dims[2] + hidden_dims[1], hidden_dims[1])

        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv_up2 = DIPBlock(hidden_dims[1] + hidden_dims[0], hidden_dims[0])

        self.outc = nn.Sequential(
            nn.Conv2d(hidden_dims[0], out_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(z)
        x2 = self.down1(x1)
        x3 = self.down2(x2)

        x = self.up1(x3)
        if x.shape[2:] != x2.shape[2:]:
            x = F.interpolate(x, size=x2.shape[2:], mode="bilinear", align_corners=True)
        x = self.conv_up1(torch.cat([x2, x], dim=1))

        x = self.up2(x)
        if x.shape[2:] != x1.shape[2:]:
            x = F.interpolate(x, size=x1.shape[2:], mode="bilinear", align_corners=True)
        x = self.conv_up2(torch.cat([x1, x], dim=1))

        return self.outc(x)


def total_variation_loss(x: torch.Tensor) -> torch.Tensor:
    """Computes Total Variation (TV) regularizer to promote spatial smoothness."""
    tv_h = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean()
    tv_w = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()
    return tv_h + tv_w


class DIPInpainter:
    """Test-Time Training-Free DIP inpainter for satellite scenes."""

    def __init__(
        self,
        z_channels: int = 8,
        out_channels: int = 3,
        device: Optional[str] = None,
    ):
        self.z_channels = z_channels
        self.out_channels = out_channels
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def inpaint(
        self,
        cloudy: Union[torch.Tensor, np.ndarray],
        cloud_mask: Union[torch.Tensor, np.ndarray],
        num_iters: int = 100,
        lr: float = 0.01,
        tv_weight: float = 1e-4,
    ) -> torch.Tensor:
        """Inpaint occluded / corrupted cloud regions using test-time optimization.

        Args:
            cloudy: Input cloudy image (B, C, H, W) or (C, H, W) in [0, 1]
            cloud_mask: Binary or probability mask where 1=cloud/corrupted, 0=clear
            num_iters: Number of optimization iterations (e.g. 50-100)
            lr: Adam learning rate
            tv_weight: Weight for Total Variation regularizer

        Returns:
            Inpainted clear optical tensor in [0, 1]
        """
        # Format conversion to tensor
        if isinstance(cloudy, np.ndarray):
            if cloudy.ndim == 3 and cloudy.shape[-1] in (3, 4):
                cloudy = np.transpose(cloudy, (2, 0, 1))
            cloudy_t = torch.from_numpy(cloudy).float()
        else:
            cloudy_t = cloudy.float()

        if cloudy_t.ndim == 3:
            cloudy_t = cloudy_t.unsqueeze(0)

        if isinstance(cloud_mask, np.ndarray):
            if cloud_mask.ndim == 3 and cloud_mask.shape[-1] == 1:
                cloud_mask = cloud_mask.squeeze(-1)
            mask_t = torch.from_numpy(cloud_mask).float()
        else:
            mask_t = cloud_mask.float()

        if mask_t.ndim == 2:
            mask_t = mask_t.unsqueeze(0).unsqueeze(0)
        elif mask_t.ndim == 3:
            mask_t = mask_t.unsqueeze(1)

        cloudy_t = cloudy_t.to(self.device)
        mask_t = mask_t.to(self.device)

        if mask_t.shape[1] != cloudy_t.shape[1]:
            mask_t = mask_t.expand(-1, cloudy_t.shape[1], -1, -1)

        clear_mask = 1.0 - mask_t
        b, c, h, w = cloudy_t.shape

        # Initialize network and fixed meshgrid/noise tensor
        net = DIPNet(in_channels=self.z_channels, out_channels=c).to(self.device)
        net.train()

        z = torch.randn(b, self.z_channels, h, w, device=self.device)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)

        for _ in range(num_iters):
            optimizer.zero_grad()
            pred = net(z)

            # Fit strictly to clear pixels
            rec_loss = (torch.abs(pred - cloudy_t) * clear_mask).sum() / (clear_mask.sum() + 1e-7)
            tv_loss = total_variation_loss(pred)
            loss = rec_loss + tv_weight * tv_loss

            loss.backward()
            optimizer.step()

        net.eval()
        with torch.no_grad():
            inpainted = net(z)

        # Composite output: clear pixels from original observation, occluded from DIP
        output = cloudy_t * clear_mask + inpainted * mask_t
        return torch.clamp(output, 0.0, 1.0)

    @staticmethod
    def blend_fallback(
        reconstructed: torch.Tensor,
        dip_result: torch.Tensor,
        fallback_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Blend primary model reconstruction with DIP fallback based on reliability mask."""
        if fallback_mask.ndim == 3:
            fallback_mask = fallback_mask.unsqueeze(1)
        if fallback_mask.shape[1] != reconstructed.shape[1]:
            fallback_mask = fallback_mask.expand_as(reconstructed)

        mask = fallback_mask.float().clamp(0.0, 1.0)
        blended = (1.0 - mask) * reconstructed + mask * dip_result
        return torch.clamp(blended, 0.0, 1.0)
