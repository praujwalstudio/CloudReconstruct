"""Sunlight Direction-Aware Augmentation (SDAA)
==============================================
Physically-based augmentation module to simulate realistic cloud shadow boundaries
and sunlight direction interactions on multi-spectral satellite imagery.

Mathematical Formulation:
1. Solar Vector Shift:
   y_sh = y_cl + r * sin(theta_Z) * cos(theta_A + theta_AO)
   x_sh = x_cl + r * sin(theta_Z) * sin(theta_A + theta_AO)
   where theta_Z is solar zenith, theta_A is solar azimuth, theta_AO is azimuth offset,
   and r is cloud altitude factor.
2. Gamma Transformation for Optical Shadow Depth:
   i' = i ^ gamma (where gamma > 1.0 reduces radiance in shadowed terrain)
"""

import math
from typing import Optional, Union, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def local_histogram_shadow_removal(
    image: torch.Tensor,
    shadow_mask: torch.Tensor,
    kernel_size: int = 15,
) -> torch.Tensor:
    """Softens/removes pre-existing shadows using local mean/variance matching."""
    # Compute local mean in lit vs shadowed areas
    pad = kernel_size // 2
    b, c, h, w = image.shape
    kernel = torch.ones(1, 1, kernel_size, kernel_size, device=image.device) / (kernel_size * kernel_size)
    kernel = kernel.expand(c, 1, kernel_size, kernel_size)

    local_mean = F.conv2d(image, kernel, padding=pad, groups=c)
    lit_mask = 1.0 - shadow_mask.float()
    
    # Scale shadowed pixels toward local lit illumination
    boost_factor = torch.clamp(local_mean / (image + 1e-4), 1.0, 2.5)
    adjusted_shadow = image * boost_factor
    return image * lit_mask + adjusted_shadow * shadow_mask.float()


def apply_sdaa(
    image: torch.Tensor,
    cloud_mask: torch.Tensor,
    solar_zenith_rad: Optional[Union[float, torch.Tensor]] = None,
    solar_azimuth_rad: Optional[Union[float, torch.Tensor]] = None,
    r_height: float = 15.0,
    azimuth_offset_rad: float = 0.0,
    gamma: float = 2.0,
    blur_sigma: float = 1.2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applies Sunlight Direction-Aware Augmentation (SDAA).

    Args:
        image: Optical satellite image tensor of shape (B, C, H, W) in [0, 1].
        cloud_mask: Cloud mask tensor of shape (B, 1, H, W) or (B, H, W) in [0, 1].
        solar_zenith_rad: Solar zenith angle in radians (sampled if None).
        solar_azimuth_rad: Solar azimuth angle in radians (sampled if None).
        r_height: Cloud altitude/height scale factor (in pixel units).
        azimuth_offset_rad: Solar azimuth offset in radians.
        gamma: Non-linear shadow depth exponent (i' = i^gamma).
        blur_sigma: Gaussian kernel blur to model soft penumbra shadow boundaries.

    Returns:
        augmented_image: Image with synthesized direction-aware shadows.
        shadow_mask: 2D binary/continuous shadow mask tensor.
    """
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if cloud_mask.ndim == 3:
        cloud_mask = cloud_mask.unsqueeze(1)

    b, c, h, w = image.shape
    device = image.device

    # Sample solar angles if not provided (Zenith between 15° and 65°, Azimuth 0° to 360°)
    if solar_zenith_rad is None:
        zenith = torch.rand(b, 1, 1, 1, device=device) * math.radians(50) + math.radians(15)
    elif isinstance(solar_zenith_rad, (float, int)):
        zenith = torch.full((b, 1, 1, 1), float(solar_zenith_rad), device=device)
    else:
        zenith = solar_zenith_rad.view(b, 1, 1, 1).to(device)

    if solar_azimuth_rad is None:
        azimuth = torch.rand(b, 1, 1, 1, device=device) * (2 * math.pi)
    elif isinstance(solar_azimuth_rad, (float, int)):
        azimuth = torch.full((b, 1, 1, 1), float(solar_azimuth_rad), device=device)
    else:
        azimuth = solar_azimuth_rad.view(b, 1, 1, 1).to(device)

    total_azimuth = azimuth + azimuth_offset_rad

    # 2D shift along solar vector:
    # y_sh = y_cl + r * sin(theta_Z) * cos(theta_A + theta_AO)
    # x_sh = x_cl + r * sin(theta_Z) * sin(theta_A + theta_AO)
    shift_y = r_height * torch.sin(zenith) * torch.cos(total_azimuth)
    shift_x = r_height * torch.sin(zenith) * torch.sin(total_azimuth)

    # Normalize shifts to affine grid [-1, 1] range: dx_norm = 2 * shift_x / w
    dx_norm = 2.0 * shift_x / float(w)
    dy_norm = 2.0 * shift_y / float(h)

    # Construct affine transformation matrix [ [1, 0, -dx_norm], [0, 1, -dy_norm] ]
    theta = torch.zeros(b, 2, 3, device=device)
    theta[:, 0, 0] = 1.0
    theta[:, 1, 1] = 1.0
    theta[:, 0, 2] = -dx_norm.squeeze()
    theta[:, 1, 2] = -dy_norm.squeeze()

    grid = F.affine_grid(theta, cloud_mask.size(), align_corners=True)
    shifted_shadow = F.grid_sample(cloud_mask.float(), grid, mode="bilinear", padding_mode="zeros", align_corners=True)

    # Soften shadow boundary (penumbra modeling)
    if blur_sigma > 0:
        k_size = 5
        k = torch.arange(k_size, dtype=torch.float32, device=device) - k_size // 2
        gauss = torch.exp(-(k ** 2) / (2 * blur_sigma ** 2))
        gauss = gauss / gauss.sum()
        k_2d = (gauss[:, None] * gauss[None, :]).unsqueeze(0).unsqueeze(0)
        shifted_shadow = F.conv2d(shifted_shadow, k_2d, padding=k_size // 2)

    # Remove overlaps with the cloud itself so shadow is projected only on ground
    ground_shadow = torch.clamp(shifted_shadow - cloud_mask.float(), 0.0, 1.0)

    # Apply gamma brightness transformation to simulate depth and absorption:
    # i' = i^gamma in shadowed regions, linearly blended by shadow strength
    shadow_depth = torch.clamp(ground_shadow, 0.0, 1.0)
    darkened = torch.pow(torch.clamp(image, 1e-4, 1.0), gamma)
    augmented_image = (1.0 - shadow_depth) * image + shadow_depth * darkened

    return augmented_image, ground_shadow


class SunlightDirectionAwareAugmentation(nn.Module):
    """PyTorch Module for Sunlight Direction-Aware Augmentations (SDAA)."""

    def __init__(
        self,
        p: float = 0.5,
        r_range: Tuple[float, float] = (5.0, 25.0),
        gamma_range: Tuple[float, float] = (1.5, 2.5),
        solar_zenith: Optional[float] = None,
        solar_azimuth: Optional[float] = None,
    ):
        super().__init__()
        self.p = p
        self.r_range = r_range
        self.gamma_range = gamma_range
        self.solar_zenith = solar_zenith
        self.solar_azimuth = solar_azimuth

    def forward(
        self,
        image: torch.Tensor,
        cloud_mask: torch.Tensor,
        solar_zenith: Optional[float] = None,
        solar_azimuth: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.training or torch.rand(1).item() > self.p:
            return image, torch.zeros_like(cloud_mask)

        r = float(np.random.uniform(self.r_range[0], self.r_range[1]))
        gamma = float(np.random.uniform(self.gamma_range[0], self.gamma_range[1]))
        zenith = solar_zenith if solar_zenith is not None else self.solar_zenith
        azimuth = solar_azimuth if solar_azimuth is not None else self.solar_azimuth

        return apply_sdaa(
            image=image,
            cloud_mask=cloud_mask,
            solar_zenith_rad=zenith,
            solar_azimuth_rad=azimuth,
            r_height=r,
            gamma=gamma,
        )
