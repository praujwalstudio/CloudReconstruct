import numpy as np
from typing import Optional, Union


def compute_confidence(
    density: np.ndarray,
    temporal_variance: np.ndarray = None,
    sar_coherence: np.ndarray = None,
    terrain_shadow: np.ndarray = None,
    uncertainty: np.ndarray = None,
    calibrated_variance: np.ndarray = None,
) -> np.ndarray:
    """Computes a pixel-wise composite confidence map in [0, 1].

    Args:
        density: Cloud density / cloud probability map in [0, 1]
        temporal_variance: Empirical variance across temporal acquisitions
        sar_coherence: SAR spatial coherence map
        terrain_shadow: Topographic shadow occlusion mask
        uncertainty: Model predictive uncertainty map
        calibrated_variance: Single-pass heteroscedastic variance (sigma^2)
    """
    confidence = 1.0 - density.copy()

    if temporal_variance is not None:
        norm_var = np.clip(temporal_variance / (temporal_variance.max() + 1e-8), 0, 1)
        temporal_conf = 1.0 - norm_var
        confidence = 0.6 * confidence + 0.4 * temporal_conf

    if sar_coherence is not None:
        confidence = 0.7 * confidence + 0.3 * sar_coherence

    if terrain_shadow is not None:
        confidence = confidence * (1.0 - 0.5 * terrain_shadow)

    if uncertainty is not None:
        norm_unc = np.clip(uncertainty / (uncertainty.max() + 1e-8), 0, 1)
        confidence = confidence * (1.0 - 0.3 * norm_unc)

    if calibrated_variance is not None:
        # Exponential confidence decay based on calibrated heteroscedastic variance
        # When variance -> 0, conf -> 1.0; when variance is high, conf -> 0.0
        var_f = np.nan_to_num(calibrated_variance.astype(np.float32), nan=1.0)
        calibrated_conf = np.exp(-var_f / 0.1)
        confidence = 0.5 * confidence + 0.5 * np.clip(calibrated_conf, 0.0, 1.0)

    return np.clip(confidence, 0.0, 1.0)


def aggregate_confidence(confidence_map: np.ndarray, patch_size: int = 64) -> float:
    return float(np.mean(confidence_map))


def generate_fallback_mask(confidence_map: np.ndarray, threshold: float = 0.4) -> np.ndarray:
    """Returns binary mask where 1 indicates low confidence requiring fallback inpainting."""
    return (confidence_map < threshold).astype(np.uint8)


class ConfidenceMap:
    def __init__(self):
        self.map = None

    def compute(
        self,
        density: np.ndarray,
        temporal_variance: np.ndarray = None,
        sar_coherence: np.ndarray = None,
        terrain_shadow: np.ndarray = None,
        uncertainty: np.ndarray = None,
        calibrated_variance: np.ndarray = None,
    ) -> np.ndarray:
        self.map = compute_confidence(
            density,
            temporal_variance,
            sar_coherence,
            terrain_shadow,
            uncertainty,
            calibrated_variance,
        )
        return self.map

    def aggregate(self, patch_size: int = 64) -> float:
        if self.map is None:
            return 0.0
        return aggregate_confidence(self.map, patch_size)

    def threshold_mask(self, threshold: float = 0.5) -> np.ndarray:
        if self.map is None:
            return np.array(0, dtype=np.uint8)
        return (self.map >= threshold).astype(np.uint8)

    def fallback_mask(self, threshold: float = 0.4) -> np.ndarray:
        if self.map is None:
            return np.array(0, dtype=np.uint8)
        return generate_fallback_mask(self.map, threshold)

