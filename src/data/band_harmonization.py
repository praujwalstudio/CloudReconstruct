import numpy as np
import torch
from typing import Union

# S2 L1C/L2A Band Indices:
# B1:0, B2:1, B3(Green):2, B4(Red):3, B5:4, B6:5, B7:6, B8(NIR):7, B8A:8, B9:9, B10:10, B11:11, B12:12
S2_TO_LISS4_INDICES = [2, 3, 7]  # Green (B3), Red (B4), NIR (B8)
TOA_SCALE_FACTOR = 10000.0


def harmonize_s2_to_liss4(s2_data: Union[torch.Tensor, np.ndarray],
                          scale_toa: bool = True,
                          clip_range: tuple[float, float] = (0.0, 1.0)) -> Union[torch.Tensor, np.ndarray]:
    """Harmonizes 13-band Sentinel-2 data down to 3-channel LISS-IV equivalent (Green, Red, NIR).

    Args:
        s2_data: Sentinel-2 tensor (B, 13, H, W) or (13, H, W) or numpy array (H, W, 13) or (13, H, W).
        scale_toa: If True, divide by 10,000.0 when maximum value > 1.0 to convert DN to reflectance.
        clip_range: (min, max) range to clip reflectance values (default: 0.0 to 1.0).

    Returns:
        3-channel tensor or array corresponding to LISS-IV [Green, Red, NIR].
    """
    if isinstance(s2_data, torch.Tensor):
        # PyTorch Tensor
        if s2_data.ndim == 4:  # (B, C, H, W)
            if s2_data.shape[1] < 8:
                raise ValueError(f"Expected at least 8 or 13 channels for S2 tensor, got shape {s2_data.shape}")
            liss4 = s2_data[:, S2_TO_LISS4_INDICES, :, :]
        elif s2_data.ndim == 3:  # (C, H, W)
            if s2_data.shape[0] < 8:
                raise ValueError(f"Expected at least 8 or 13 channels for S2 tensor, got shape {s2_data.shape}")
            liss4 = s2_data[S2_TO_LISS4_INDICES, :, :]
        else:
            raise ValueError(f"Unsupported tensor shape: {s2_data.shape}")

        liss4 = liss4.float()
        if scale_toa:
            if liss4.max() > 1.0 or not torch.is_floating_point(s2_data):
                liss4 = liss4 / TOA_SCALE_FACTOR
        if clip_range is not None:
            liss4 = torch.clamp(liss4, clip_range[0], clip_range[1])
        return liss4

    elif isinstance(s2_data, np.ndarray):
        # NumPy Array
        if s2_data.ndim == 3:
            if 8 <= s2_data.shape[0] <= 16:  # (13, H, W)
                liss4 = s2_data[S2_TO_LISS4_INDICES, :, :].astype(np.float32)
            elif s2_data.shape[2] >= 8:  # (H, W, 13)
                liss4 = s2_data[:, :, S2_TO_LISS4_INDICES].astype(np.float32)
            else:
                raise ValueError(f"Expected 13 channels in axis 0 or 2, got shape {s2_data.shape}")
        elif s2_data.ndim == 4 and s2_data.shape[1] >= 8:  # (B, 13, H, W)
            liss4 = s2_data[:, S2_TO_LISS4_INDICES, :, :].astype(np.float32)
        else:
            raise ValueError(f"Unsupported array shape: {s2_data.shape}")

        if scale_toa:
            if liss4.max() > 1.0 or s2_data.dtype.kind in ('i', 'u'):
                liss4 = liss4 / TOA_SCALE_FACTOR
        if clip_range is not None:
            liss4 = np.clip(liss4, clip_range[0], clip_range[1])
        return liss4

    else:
        raise TypeError(f"Expected torch.Tensor or np.ndarray, got {type(s2_data)}")
