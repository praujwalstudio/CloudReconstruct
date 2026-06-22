import numpy as np
from pathlib import Path


def load_dem(dem_path: Path) -> tuple[np.ndarray, dict]:
    import rasterio
    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype(np.float32)
        profile = src.profile
    return dem, profile


def compute_slope_aspect(dem: np.ndarray, resolution: float = 5.8) -> tuple[np.ndarray, np.ndarray]:
    gy, gx = np.gradient(dem, resolution, resolution)
    slope = np.arctan(np.sqrt(gx**2 + gy**2))
    aspect = np.arctan2(-gy, gx)
    aspect = np.where(aspect < 0, aspect + 2 * np.pi, aspect)
    return slope, aspect


def cosine_correction(image: np.ndarray, slope: np.ndarray, aspect: np.ndarray,
                      sun_zenith: float, sun_azimuth: float) -> np.ndarray:
    cos_i = np.cos(sun_zenith) * np.cos(slope) + \
            np.sin(sun_zenith) * np.sin(slope) * np.cos(sun_azimuth - aspect)
    cos_i = np.clip(cos_i, 0.01, 1.0)
    cos_theta = np.cos(sun_zenith)
    factor = cos_theta / cos_i
    if image.ndim == 3 and factor.ndim == 2:
        factor = factor[..., np.newaxis]
    corrected = image.astype(np.float32) * factor
    return np.clip(corrected, 0, None).astype(image.dtype)


def c_correction(image: np.ndarray, slope: np.ndarray, aspect: np.ndarray,
                 sun_zenith: float, sun_azimuth: float) -> np.ndarray:
    cos_i = np.cos(sun_zenith) * np.cos(slope) + \
            np.sin(sun_zenith) * np.sin(slope) * np.cos(sun_azimuth - aspect)
    cos_i = np.clip(cos_i, 0.01, 1.0)
    cos_theta = np.cos(sun_zenith)

    flat = image.reshape(image.shape[0], -1)
    ci_flat = cos_i.reshape(1, -1)
    c_vals = []
    for band in flat:
        mask = ci_flat[0] > 0
        if mask.sum() < 10:
            c_vals.append(0.0)
            continue
        A = np.column_stack([ci_flat[0, mask], np.ones(mask.sum())])
        b = band[mask]
        coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        c = coeffs[1] / coeffs[0] if abs(coeffs[0]) > 1e-8 else 0.0
        c_vals.append(max(c, 0.0))

    corrected = image.astype(np.float32)
    for i, c in enumerate(c_vals):
        band_arr = corrected[i] if corrected.ndim == 3 else corrected
        if c > 0:
            factor = (cos_theta + c) / (cos_i + c)
            if band_arr.ndim == 2 and factor.ndim == 2:
                pass
            band_arr[...] = band_arr * factor
        else:
            band_arr[...] = band_arr

    return np.clip(corrected, 0, None).astype(image.dtype)


def detect_terrain_shadows(dem: np.ndarray, slope: np.ndarray, aspect: np.ndarray,
                           sun_zenith: float, sun_azimuth: float,
                           threshold: float = 0.2) -> np.ndarray:
    cos_i = np.cos(sun_zenith) * np.cos(slope) + \
            np.sin(sun_zenith) * np.sin(slope) * np.cos(sun_azimuth - aspect)
    direct_illum = cos_i > threshold
    hillshade = (cos_i > 0).astype(np.float32)

    from scipy.ndimage import uniform_filter
    shadow_buffer = 1.0 - uniform_filter(direct_illum.astype(np.float32), size=5)
    return (shadow_buffer > 0.3).astype(np.uint8)


class TerrainProcessor:
    def __init__(self, resolution: float = 5.8):
        self.resolution = resolution
        self.dem = None
        self.slope = None
        self.aspect = None

    def load(self, dem_path: Path):
        self.dem, self.profile = load_dem(dem_path)
        self.slope, self.aspect = compute_slope_aspect(self.dem, self.resolution)
        return self

    def correct(self, image: np.ndarray, sun_zenith: float = np.radians(45),
                sun_azimuth: float = np.radians(180), method: str = "cosine") -> np.ndarray:
        if self.slope is None or self.aspect is None:
            raise ValueError("Call load() first to load DEM data")
        if method == "cosine":
            return cosine_correction(image, self.slope, self.aspect, sun_zenith, sun_azimuth)
        elif method == "c-correction":
            return c_correction(image, self.slope, self.aspect, sun_zenith, sun_azimuth)
        else:
            raise ValueError(f"Unknown correction method: {method}")

    def shadow_mask(self, sun_zenith: float = np.radians(45),
                    sun_azimuth: float = np.radians(180)) -> np.ndarray:
        if self.slope is None or self.aspect is None:
            raise ValueError("Call load() first to load DEM data")
        return detect_terrain_shadows(self.dem, self.slope, self.aspect,
                                      sun_zenith, sun_azimuth)
