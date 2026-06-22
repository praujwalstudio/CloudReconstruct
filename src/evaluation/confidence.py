import numpy as np


def compute_confidence(density: np.ndarray, temporal_variance: np.ndarray = None,
                       sar_coherence: np.ndarray = None,
                       terrain_shadow: np.ndarray = None,
                       uncertainty: np.ndarray = None) -> np.ndarray:
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

    return np.clip(confidence, 0, 1)


def aggregate_confidence(confidence_map: np.ndarray, patch_size: int = 64) -> float:
    return float(np.mean(confidence_map))


class ConfidenceMap:
    def __init__(self):
        self.map = None

    def compute(self, density: np.ndarray, temporal_variance: np.ndarray = None,
                sar_coherence: np.ndarray = None,
                terrain_shadow: np.ndarray = None,
                uncertainty: np.ndarray = None) -> np.ndarray:
        self.map = compute_confidence(density, temporal_variance, sar_coherence,
                                      terrain_shadow, uncertainty)
        return self.map

    def aggregate(self, patch_size: int = 64) -> float:
        if self.map is None:
            return 0.0
        return aggregate_confidence(self.map, patch_size)

    def threshold_mask(self, threshold: float = 0.5) -> np.ndarray:
        if self.map is None:
            return np.array(0, dtype=np.uint8)
        return (self.map >= threshold).astype(np.uint8)
