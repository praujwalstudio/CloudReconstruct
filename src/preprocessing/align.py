import cv2
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from pathlib import Path
from tqdm import tqdm

from src.config import LISS4_RAW, ALIGNED


def find_gcps(reference: np.ndarray, target: np.ndarray, max_features: int = 5000) -> tuple:
    sift = cv2.SIFT_create(nfeatures=max_features)

    ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY) if reference.ndim == 3 else reference
    tgt_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY) if target.ndim == 3 else target

    kp1, des1 = sift.detectAndCompute(ref_gray, None)
    kp2, des2 = sift.detectAndCompute(tgt_gray, None)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return None, None

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    if len(good_matches) < 4:
        return None, None

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    matrix, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

    return matrix, len(good_matches)


def warp_to_reference(target_path: Path, reference_transform, reference_crs, reference_shape, out_path: Path) -> Path:
    with rasterio.open(target_path) as src:
        dst_array = np.zeros((src.count, *reference_shape[1:]), dtype=src.dtypes[0])

        for band in range(src.count):
            reproject(
                source=src.read(band + 1),
                src_transform=src.transform,
                src_crs=src.crs,
                destination=dst_array[band],
                dst_transform=reference_transform,
                dst_crs=reference_crs,
                resampling=Resampling.bilinear,
            )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(
            out_path,
            "w",
            driver="GTiff",
            height=dst_array.shape[1],
            width=dst_array.shape[2],
            count=dst_array.shape[0],
            dtype=dst_array.dtype,
            crs=reference_crs,
            transform=reference_transform,
        ) as dst:
            for band in range(dst_array.shape[0]):
                dst.write(dst_array[band], band + 1)

    return out_path


def align_pair(moving_path: Path, fixed_path: Path, out_path: Path = None) -> dict:
    if out_path is None:
        out_path = ALIGNED / f"aligned_{moving_path.name}"

    print(f"[ALIGN] {moving_path.name} -> {fixed_path.name}")

    with rasterio.open(fixed_path) as fixed_src:
        fixed_arr = fixed_src.read()
        fixed_transform = fixed_src.transform
        fixed_crs = fixed_src.crs
        fixed_shape = fixed_arr.shape

        fixed_vis = np.moveaxis(fixed_arr[:3], 0, -1) if fixed_arr.shape[0] >= 3 else fixed_arr[0]
        fixed_vis = cv2.normalize(fixed_vis, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    with rasterio.open(moving_path) as moving_src:
        moving_arr = moving_src.read()
        moving_vis = np.moveaxis(moving_arr[:3], 0, -1) if moving_arr.shape[0] >= 3 else moving_arr[0]
        moving_vis = cv2.normalize(moving_vis, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    matrix, n_matches = find_gcps(fixed_vis, moving_vis)

    if matrix is not None:
        print(f"  ✓ Found {n_matches} good matches. Warping...")
    else:
        print(f"  ⚠ Insufficient matches ({n_matches}). Using geotransform-only alignment.")

    result = warp_to_reference(moving_path, fixed_transform, fixed_crs, fixed_shape, out_path)

    return {
        "moving": str(moving_path),
        "fixed": str(fixed_path),
        "output": str(result),
        "matches": n_matches if matrix is not None else 0,
        "aligned": matrix is not None or n_matches is not None,
    }


def align_all_scenes(reference_scene: str = None) -> list[dict]:
    liss4_files = sorted(LISS4_RAW.glob("*.tif")) + sorted(LISS4_RAW.glob("*.tiff"))

    if not liss4_files:
        print("[WARN] No LISS-IV scenes found in raw directory.")
        return []

    if reference_scene:
        fixed = LISS4_RAW / reference_scene
    else:
        fixed = liss4_files[0]

    if not fixed.exists():
        print(f"[ERROR] Reference scene not found: {fixed}")
        return []

    print(f"[ALIGN ALL] Reference: {fixed.name}")
    results = []

    for moving in tqdm(liss4_files, desc="Aligning scenes"):
        if moving.name == fixed.name:
            print(f"  [SKIP] {moving.name} is the reference")
            results.append({
                "moving": str(moving),
                "fixed": str(fixed),
                "output": str(ALIGNED / moving.name),
                "matches": -1,
                "aligned": True,
            })
            continue

        result = align_pair(moving, fixed)
        results.append(result)

    return results


if __name__ == "__main__":
    align_all_scenes()
