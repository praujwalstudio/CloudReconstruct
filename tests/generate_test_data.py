"""
Generate synthetic LISS-IV-like GeoTIFF scenes for end-to-end pipeline testing.
Creates small (512x512) test scenes with controlled cloud patterns.

Usage:
    python tests/generate_test_data.py
"""

import os
import sys
from pathlib import Path

# Fix PROJ database conflict with PostGIS
_proj_lib = Path(r"C:\Users\COIN\AppData\Local\Programs\Python\Python311\Lib\site-packages\rasterio\proj_data")
if _proj_lib.exists():
    os.environ["PROJ_LIB"] = str(_proj_lib)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import rasterio
from rasterio.transform import from_origin
from src.config import LISS4_RAW, DEM_RAW


def make_band():
    h, w = 512, 512
    # Base intensity with some texture
    band = np.random.randint(150, 350, (h, w)).astype(np.uint16)

    # Add a geometric feature (rectangle)
    band[100:200, 100:200] = 500
    band[300:400, 350:450] = 100

    # Add a linear feature (road-like)
    band[250, 100:400] = 600

    return band


def make_cloud_mask(h=512, w=512):
    mask = np.zeros((h, w), dtype=np.uint8)

    # Thin cloud — semi-transparent region
    mask[50:150, 50:250] = 1

    # Medium cloud
    mask[200:350, 100:300] = 2

    # Dense cloud
    mask[150:300, 350:450] = 3

    return mask


def apply_cloud_to_image(image, cloud_mask):
    cloudy = image.copy().astype(np.float32)

    for y in range(cloudy.shape[0]):
        for x in range(cloudy.shape[1]):
            density = cloud_mask[y, x]
            if density == 1:  # thin
                cloudy[y, x] = cloudy[y, x] * 0.6 + 400 * 0.4
            elif density == 2:  # medium
                cloudy[y, x] = cloudy[y, x] * 0.3 + 600 * 0.7
            elif density == 3:  # dense
                cloudy[y, x] = 800

    return cloudy.astype(np.uint16)


def generate_test_scene(out_path: Path, scene_id: str, clear: bool = False):
    """Generate a single LISS-IV-like GeoTIFF (3 bands: G, R, NIR)."""
    h, w = 512, 512

    green = make_band()
    red = make_band()
    nir = make_band() + 200  # NIR generally higher

    image = np.stack([green, red, nir], axis=0)

    if not clear:
        cloud_mask = make_cloud_mask()
        for band in range(3):
            image[band] = apply_cloud_to_image(image[band], cloud_mask)

    profile = {
        "driver": "GTiff",
        "height": h,
        "width": w,
        "count": 3,
        "dtype": "uint16",
        "crs": "EPSG:4326",
        "transform": from_origin(89.0, 26.0, 0.0005, 0.0005),  # ~50m per pixel
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(image)

    print(f"  [OK] Created {out_path.name} (clear={clear})")
    return out_path


def generate_dem(out_path: Path):
    h, w = 512, 512
    # Simple gradient DEM — low in south, high in north (NE India pattern)
    dem = np.linspace(100, 1500, h).reshape(h, 1) + np.random.randint(-10, 10, (h, w))

    profile = {
        "driver": "GTiff",
        "height": h,
        "width": w,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": from_origin(89.0, 26.0, 0.0005, 0.0005),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(dem.astype(np.float32), 1)

    print(f"  [OK] Created DEM: {out_path.name}")


def main():
    print("Generating synthetic test data...\n")

    # Scene 1: Cloudy (the one we want to reconstruct)
    generate_test_scene(LISS4_RAW / "test_scene_cloudy.tif", "test_cloudy", clear=False)

    # Scene 2: Clear reference (same location, different date)
    generate_test_scene(LISS4_RAW / "test_scene_clear.tif", "test_clear", clear=True)

    # Scene 3: Another cloudy scene for multi-temporal
    generate_test_scene(LISS4_RAW / "test_scene_cloudy2.tif", "test_cloudy2", clear=False)

    # Scene 4: Another clear
    generate_test_scene(LISS4_RAW / "test_scene_clear2.tif", "test_clear2", clear=True)

    # DEM
    generate_dem(DEM_RAW / "test_dem.tif")

    print(f"\nDone. Test data saved to: {LISS4_RAW.parent}")
    print("Run `python main.py` to test the pipeline.")


if __name__ == "__main__":
    main()
