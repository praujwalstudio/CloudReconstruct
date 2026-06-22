import numpy as np
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity


def _infer_data_range(image: np.ndarray) -> float:
    if np.issubdtype(image.dtype, np.floating):
        return 1.0
    return float(np.iinfo(image.dtype).max)


def _ndvi(image: np.ndarray) -> np.ndarray:
    red = image[..., 1].astype(np.float32)
    nir = image[..., 2].astype(np.float32)
    denom = nir + red + 1e-8
    return (nir - red) / denom


def psnr(pred: np.ndarray, target: np.ndarray,
         data_range: float = None, mask: np.ndarray = None) -> float:
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: {pred.shape} vs {target.shape}")
    if data_range is None:
        data_range = _infer_data_range(target)
    diff = pred.astype(np.float32) - target.astype(np.float32)
    if mask is not None:
        if mask.ndim == 2 and diff.ndim == 3:
            mask = mask[..., np.newaxis]
        diff = diff * mask.astype(np.float32)
        mse = np.sum(diff ** 2) / (np.sum(mask) + 1e-8)
    else:
        mse = np.mean(diff ** 2)
    if mse < 1e-12:
        return float("inf")
    return float(10 * np.log10(data_range ** 2 / mse))


def sam(pred: np.ndarray, target: np.ndarray, mask: np.ndarray = None) -> float:
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: {pred.shape} vs {target.shape}")
    pred_f = pred.astype(np.float32)
    target_f = target.astype(np.float32)
    dot = np.sum(pred_f * target_f, axis=-1)
    norm_p = np.linalg.norm(pred_f, axis=-1)
    norm_t = np.linalg.norm(target_f, axis=-1)
    denom = np.maximum(norm_p * norm_t, 1e-12)
    cos_angle = np.clip(dot / denom, -1, 1)
    angle = np.arccos(cos_angle)
    if mask is not None:
        angle = angle * mask.astype(np.float32)
        return float(np.degrees(np.sum(angle) / (np.sum(mask) + 1e-8)))
    return float(np.degrees(np.mean(angle)))


def ndvi_correlation(pred: np.ndarray, target: np.ndarray,
                     mask: np.ndarray = None) -> float:
    ndvi_p = _ndvi(pred)
    ndvi_t = _ndvi(target)
    if mask is not None:
        valid = mask > 0
        if valid.sum() < 2:
            return 0.0
        ndvi_p = ndvi_p[valid]
        ndvi_t = ndvi_t[valid]
    corr, _ = pearsonr(ndvi_p.ravel(), ndvi_t.ravel())
    return float(corr) if not np.isnan(corr) else 0.0


def compute_all_metrics(pred: np.ndarray, target: np.ndarray,
                        mask: np.ndarray = None,
                        data_range: float = None) -> dict:
    if data_range is None:
        data_range = _infer_data_range(target)
    ssim_val = structural_similarity(
        pred.astype(np.float32), target.astype(np.float32),
        data_range=data_range,
        channel_axis=-1 if pred.ndim == 3 else None,
    )
    return {
        "psnr": psnr(pred, target, data_range, mask),
        "ssim": ssim_val,
        "sam": sam(pred, target, mask),
        "ndvi_correlation": ndvi_correlation(pred, target, mask),
    }
