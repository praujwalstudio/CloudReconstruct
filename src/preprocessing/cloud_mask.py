import numpy as np
import rasterio
from pathlib import Path
from skimage import morphology

from src.config import ALIGNED, CLOUD_MASKS, LISS4_RAW


def compute_cloud_mask_ndvi(image: np.ndarray, threshold: float = 0.2) -> np.ndarray:
    red = image[..., 1].astype(np.float32)
    nir = image[..., 2].astype(np.float32)

    denom = nir + red + 1e-8
    ndvi = (nir - red) / denom

    return (ndvi < threshold).astype(np.uint8)


def compute_cloud_mask_brightness(image: np.ndarray, percentile: int = 85) -> np.ndarray:
    gray = np.mean(image[..., :3], axis=-1)
    threshold = np.percentile(gray, percentile)
    return (gray > threshold).astype(np.uint8)


def compute_cloud_mask_whiteness(image: np.ndarray, threshold: float = 0.1) -> np.ndarray:
    bands = image[..., :3].astype(np.float32)
    mean_band = np.mean(bands, axis=-1)
    whitened = np.abs(bands[..., 0] - mean_band) < threshold * mean_band + 1e-8
    whitened = whitened & (np.abs(bands[..., 1] - mean_band) < threshold * mean_band + 1e-8)
    whitened = whitened & (np.abs(bands[..., 2] - mean_band) < threshold * mean_band + 1e-8)
    return (whitened & (mean_band > 0.3)).astype(np.uint8)


def compute_cloud_mask_temporal(image: np.ndarray, reference: np.ndarray, threshold: float = 0.15) -> np.ndarray:
    diff = np.mean(np.abs(image.astype(np.float32) - reference.astype(np.float32)), axis=-1)
    norm_diff = diff / (np.mean(reference.astype(np.float32), axis=-1) + 1e-8)
    return (norm_diff > threshold).astype(np.uint8)


def ensemble_masks(masks: list[np.ndarray], method: str = "majority") -> np.ndarray:
    stack = np.stack(masks, axis=-1)
    if method == "majority":
        return (np.mean(stack, axis=-1) > 0.5).astype(np.uint8)
    elif method == "union":
        return (np.max(stack, axis=-1) > 0).astype(np.uint8)
    elif method == "intersection":
        return (np.min(stack, axis=-1) > 0).astype(np.uint8)
    else:
        raise ValueError(f"Unknown ensemble method: {method}")


def refine_mask(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    kernel = morphology.disk(kernel_size)
    mask = morphology.closing(mask, kernel)
    mask = morphology.opening(mask, kernel)
    return mask


def compute_cloud_mask(
    image: np.ndarray,
    use_ndvi: bool = True,
    use_brightness: bool = True,
    use_whiteness: bool = True,
    reference: np.ndarray = None,
    ensemble_method: str = "majority",
) -> np.ndarray:
    masks = []

    if use_ndvi:
        masks.append(compute_cloud_mask_ndvi(image))

    if use_brightness:
        masks.append(compute_cloud_mask_brightness(image))

    if use_whiteness:
        masks.append(compute_cloud_mask_whiteness(image))

    if reference is not None:
        masks.append(compute_cloud_mask_temporal(image, reference))

    if not masks:
        raise ValueError("At least one mask method must be enabled")

    mask = ensemble_masks(masks, ensemble_method)
    mask = refine_mask(mask)
    return mask


def cloud_density(mask: np.ndarray, patch_size: int = 64) -> np.ndarray:
    h, w = mask.shape
    density = np.zeros((h, w), dtype=np.float32)

    for y in range(0, h, patch_size):
        for x in range(0, w, patch_size):
            patch = mask[y:min(y + patch_size, h), x:min(x + patch_size, w)]
            density[y:y + patch.shape[0], x:x + patch.shape[1]] = np.mean(patch)

    return density


def compute_cloud_shadow_mask(image: np.ndarray, cloud_mask: np.ndarray = None,
                               nir_threshold: float = 0.15,
                               dark_threshold: float = 0.2) -> np.ndarray:
    nir = image[..., 2].astype(np.float32)
    dark_nir = (nir < nir_threshold).astype(np.uint8)

    if cloud_mask is not None:
        from scipy.ndimage import binary_dilation
        struct = np.ones((15, 15))
        dilated = binary_dilation(cloud_mask.astype(bool), struct)
        dark_nir = dark_nir & (~dilated)

    gray = np.mean(image[..., :3].astype(np.float32), axis=-1)
    dark_visible = (gray < dark_threshold).astype(np.uint8)
    shadow = (dark_nir & dark_visible).astype(np.uint8)
    return shadow


def classify_cloud_density(density: np.ndarray, shadow_mask: np.ndarray = None) -> np.ndarray:
    classes = np.zeros_like(density, dtype=np.uint8)
    if shadow_mask is not None:
        classes[shadow_mask > 0] = 4
    classes[(classes == 0) & (density < 0.3)] = 0    # clear
    classes[(classes == 0) & (density >= 0.3) & (density < 0.5)] = 1   # thin
    classes[(classes == 0) & (density >= 0.5) & (density < 0.8)] = 2   # medium
    classes[(classes == 0) & (density >= 0.8)] = 3    # thick
    return classes


def process_scene(scene_path: Path, reference_path: Path = None, out_dir: Path = None) -> dict:
    out_dir = Path(out_dir or CLOUD_MASKS)
    out_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(scene_path) as src:
        image = np.moveaxis(src.read(), 0, -1)
        profile = src.profile

    ref = None
    if reference_path and reference_path.exists():
        with rasterio.open(reference_path) as ref_src:
            ref = np.moveaxis(ref_src.read(), 0, -1)

    mask = compute_cloud_mask(image, reference=ref)
    shadow = compute_cloud_shadow_mask(image, mask)
    density = cloud_density(mask)
    classes = classify_cloud_density(density, shadow)

    mask_path = out_dir / f"mask_{scene_path.stem}.tif"
    profile.update(count=3, dtype=np.uint8)

    with rasterio.open(mask_path, "w", **profile) as dst:
        dst.write(mask * 255, 1)
        dst.write((density * 255).astype(np.uint8), 2)
        dst.write(classes * 64, 3)

    return {
        "scene": str(scene_path),
        "mask": str(mask_path),
        "cloud_fraction": float(np.mean(mask)),
        "density_stats": {
            "clear": float(np.mean(classes == 0)),
            "thin": float(np.mean(classes == 1)),
            "medium": float(np.mean(classes == 2)),
            "thick": float(np.mean(classes == 3)),
            "shadow": float(np.mean(classes == 4)),
        },
    }


def process_all(reference_name: str = None) -> list[dict]:
    scenes = sorted(ALIGNED.glob("*.tif")) + sorted(ALIGNED.glob("*.tiff"))

    if not scenes:
        scenes = sorted(LISS4_RAW.glob("*.tif")) + sorted(LISS4_RAW.glob("*.tiff"))

    if not scenes:
        print("[WARN] No scenes found for cloud masking.")
        return []

    ref_path = None
    if reference_name:
        for s in scenes:
            if reference_name in s.name:
                ref_path = s
                break

    results = []
    for scene in scenes:
        result = process_scene(scene, ref_path)
        results.append(result)
        density = result["density_stats"]
        print(f"  {scene.name}: cloud={result['cloud_fraction']:.2f} "
              f"[clear={density['clear']:.2f} thin={density['thin']:.2f} "
              f"medium={density['medium']:.2f} thick={density['thick']:.2f} "
              f"shadow={density['shadow']:.2f}]")

    return results


if __name__ == "__main__":
    process_all()
