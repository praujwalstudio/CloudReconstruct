import numpy as np


def ndvi(image: np.ndarray) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] < 3:
        return np.zeros(image.shape[:2], dtype=np.float32)
    red = image[..., 1].astype(np.float32)
    nir = image[..., 2].astype(np.float32)
    denom = nir + red + 1e-8
    return (nir - red) / denom


def ndvi_preservation(corrected: np.ndarray, reference: np.ndarray) -> float:
    ndvi_c = ndvi(corrected)
    ndvi_r = ndvi(reference)
    diff = np.abs(ndvi_c - ndvi_r)
    return float(1.0 - np.clip(np.mean(diff) / 2.0, 0, 1))


def structural_similarity(x: np.ndarray, y: np.ndarray) -> float:
    from skimage.metrics import structural_similarity as ssim
    if x.ndim == 3 and x.shape[-1] == 3:
        return float(ssim(x, y, channel_axis=-1, data_range=x.max() - x.min() + 1e-8))
    else:
        x_gray = np.mean(x, axis=-1) if x.ndim == 3 else x
        y_gray = np.mean(y, axis=-1) if y.ndim == 3 else y
        return float(ssim(x_gray, y_gray, data_range=x_gray.max() - x_gray.min() + 1e-8))


def compute_ars(confidence_map: np.ndarray, corrected_image: np.ndarray = None,
                reference_image: np.ndarray = None,
                weight_confidence: float = 0.4,
                weight_ndvi: float = 0.3,
                weight_structural: float = 0.3) -> dict:
    score = 0.0
    components = {}

    conf_score = float(np.mean(confidence_map))
    score += weight_confidence * conf_score
    components["confidence"] = conf_score

    ndvi_score = 0.0
    if corrected_image is not None and reference_image is not None:
        ndvi_score = ndvi_preservation(corrected_image, reference_image)
        score += weight_ndvi * ndvi_score
    components["ndvi_preservation"] = ndvi_score

    ssim_score = 0.0
    if corrected_image is not None and reference_image is not None:
        ssim_score = structural_similarity(corrected_image, reference_image)
        score += weight_structural * ssim_score
    components["structural_similarity"] = ssim_score

    components["weighted_confidence"] = weight_confidence * conf_score
    components["weighted_ndvi"] = weight_ndvi * ndvi_score
    components["weighted_structural"] = weight_structural * ssim_score

    return {
        "ars": round(score, 4),
        "components": components,
        "weights": {
            "confidence": weight_confidence,
            "ndvi": weight_ndvi,
            "structural": weight_structural,
        },
    }


class AnalysisReadiness:
    def __init__(self, weight_confidence: float = 0.4,
                 weight_ndvi: float = 0.3,
                 weight_structural: float = 0.3):
        self.weights = {
            "confidence": weight_confidence,
            "ndvi": weight_ndvi,
            "structural": weight_structural,
        }

    def evaluate(self, confidence_map: np.ndarray, corrected_image: np.ndarray = None,
                 reference_image: np.ndarray = None) -> dict:
        return compute_ars(
            confidence_map, corrected_image, reference_image,
            self.weights["confidence"], self.weights["ndvi"], self.weights["structural"],
        )

    def grade(self, ars_score: float) -> str:
        if ars_score >= 0.9:
            return "A"
        elif ars_score >= 0.75:
            return "B"
        elif ars_score >= 0.5:
            return "C"
        else:
            return "D"
