"""Topographic DEM Feature Extraction
=====================================
Computes 4-channel topographic features: [elevation, slope, aspect, hillshade]
for early fusion with multi-spectral encoder backbones.
"""

import math
from typing import Union, Tuple
import numpy as np
import torch
import torch.nn.functional as F


def compute_dem_feature_stack_np(
    dem: np.ndarray,
    resolution: float = 5.8,
    sun_zenith_deg: float = 45.0,
    sun_azimuth_deg: float = 180.0,
) -> np.ndarray:
    """Computes 4-channel DEM features from 2D elevation numpy array.

    Args:
        dem: 2D numpy array (H, W) of elevation in meters.
        resolution: Spatial pixel resolution in meters.
        sun_zenith_deg: Solar zenith in degrees.
        sun_azimuth_deg: Solar azimuth in degrees.

    Returns:
        4-channel array of shape (4, H, W) -> [elevation, slope, aspect, hillshade].
    """
    if dem.ndim == 3:
        dem = dem.squeeze(0) if dem.shape[0] == 1 else dem.squeeze(-1)

    # 1. Normalized elevation (scale to [0, 1] relative to typical elevation span)
    elev_min = dem.min()
    elev_max = dem.max()
    elev_norm = (dem - elev_min) / (elev_max - elev_min + 1e-6) if elev_max > elev_min else np.zeros_like(dem)

    # 2. Gradient, Slope & Aspect
    gy, gx = np.gradient(dem, resolution, resolution)
    slope = np.arctan(np.sqrt(gx ** 2 + gy ** 2))
    aspect = np.arctan2(-gy, gx)
    aspect = np.where(aspect < 0, aspect + 2 * np.pi, aspect)

    # 3. Hillshade
    zenith_rad = np.radians(sun_zenith_deg)
    azimuth_rad = np.radians(sun_azimuth_deg)
    cos_i = np.cos(zenith_rad) * np.cos(slope) + np.sin(zenith_rad) * np.sin(slope) * np.cos(azimuth_rad - aspect)
    hillshade = np.clip(cos_i, 0.0, 1.0)

    # Slope normalized to [0, 1] (max ~pi/2)
    slope_norm = np.clip(slope / (np.pi / 2.0), 0.0, 1.0)
    # Aspect normalized to [0, 1] (max 2*pi)
    aspect_norm = np.clip(aspect / (2 * np.pi), 0.0, 1.0)

    stack = np.stack([elev_norm, slope_norm, aspect_norm, hillshade], axis=0).astype(np.float32)
    return stack


def compute_dem_feature_stack(
    dem: Union[torch.Tensor, np.ndarray],
    resolution: float = 5.8,
    sun_zenith_deg: float = 45.0,
    sun_azimuth_deg: float = 180.0,
) -> Union[torch.Tensor, np.ndarray]:
    """Computes 4-channel DEM feature stack [elevation, slope, aspect, hillshade]."""
    if isinstance(dem, np.ndarray):
        return compute_dem_feature_stack_np(dem, resolution, sun_zenith_deg, sun_azimuth_deg)

    elif isinstance(dem, torch.Tensor):
        was_3d = False
        if dem.ndim == 3:  # (1, H, W)
            dem = dem.unsqueeze(0)
            was_3d = True
        elif dem.ndim == 2:
            dem = dem.unsqueeze(0).unsqueeze(0)
            was_3d = True

        b, _, h, w = dem.shape
        device = dem.device

        # Compute Sobel gradients for PyTorch tensors
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=device).view(1, 1, 3, 3) / (8.0 * resolution)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=device).view(1, 1, 3, 3) / (8.0 * resolution)

        gx = F.conv2d(dem, sobel_x, padding=1)
        gy = F.conv2d(dem, sobel_y, padding=1)

        slope = torch.atan(torch.sqrt(gx ** 2 + gy ** 2 + 1e-8))
        aspect = torch.atan2(-gy, gx)
        aspect = torch.where(aspect < 0, aspect + 2 * math.pi, aspect)

        zenith_rad = math.radians(sun_zenith_deg)
        azimuth_rad = math.radians(sun_azimuth_deg)

        cos_i = math.cos(zenith_rad) * torch.cos(slope) + math.sin(zenith_rad) * torch.sin(slope) * torch.cos(azimuth_rad - aspect)
        hillshade = torch.clamp(cos_i, 0.0, 1.0)

        # Normalize elevation per batch item
        elev_min = dem.view(b, 1, -1).min(dim=-1, keepdim=True)[0].view(b, 1, 1, 1)
        elev_max = dem.view(b, 1, -1).max(dim=-1, keepdim=True)[0].view(b, 1, 1, 1)
        elev_norm = (dem - elev_min) / (elev_max - elev_min + 1e-6)

        slope_norm = torch.clamp(slope / (math.pi / 2.0), 0.0, 1.0)
        aspect_norm = torch.clamp(aspect / (2 * math.pi), 0.0, 1.0)

        stack = torch.cat([elev_norm, slope_norm, aspect_norm, hillshade], dim=1)
        if was_3d and b == 1:
            stack = stack.squeeze(0)
        return stack
    else:
        raise TypeError(f"Expected torch.Tensor or np.ndarray, got {type(dem)}")
