import numpy as np


class TestComputeConfidence:
    def test_basic_from_density(self):
        from src.evaluation.confidence import compute_confidence
        density = np.array([[0.1, 0.5, 0.9]], dtype=np.float32)
        conf = compute_confidence(density)
        assert conf.shape == (1, 3)
        assert np.isclose(conf[0, 0], 0.9)
        assert np.isclose(conf[0, 1], 0.5)
        assert np.isclose(conf[0, 2], 0.1)

    def test_output_range(self):
        from src.evaluation.confidence import compute_confidence
        density = np.random.rand(50, 50).astype(np.float32)
        conf = compute_confidence(density)
        assert conf.min() >= 0.0
        assert conf.max() <= 1.0

    def test_with_temporal_variance(self):
        from src.evaluation.confidence import compute_confidence
        density = np.zeros((10, 10), dtype=np.float32)
        temporal_var = np.ones((10, 10), dtype=np.float32)
        conf = compute_confidence(density, temporal_variance=temporal_var)
        assert np.allclose(conf, 0.6, atol=0.01)

    def test_with_sar_coherence(self):
        from src.evaluation.confidence import compute_confidence
        density = np.zeros((10, 10), dtype=np.float32)
        sar_conf = np.ones((10, 10), dtype=np.float32) * 0.5
        conf = compute_confidence(density, sar_coherence=sar_conf)
        assert np.allclose(conf, 0.85, atol=0.01)

    def test_with_terrain_shadow(self):
        from src.evaluation.confidence import compute_confidence
        density = np.zeros((10, 10), dtype=np.float32)
        shadow = np.ones((10, 10), dtype=np.float32)
        conf = compute_confidence(density, terrain_shadow=shadow)
        assert np.allclose(conf, 0.5)

    def test_all_sources(self):
        from src.evaluation.confidence import compute_confidence
        density = np.full((10, 10), 0.3, dtype=np.float32)
        temporal_var = np.full((10, 10), 0.2, dtype=np.float32)
        sar_conf = np.full((10, 10), 0.8, dtype=np.float32)
        shadow = np.full((10, 10), 0.0, dtype=np.float32)
        conf = compute_confidence(density, temporal_var, sar_conf, shadow)
        assert conf.shape == (10, 10)
        assert conf.min() >= 0.0
        assert conf.max() <= 1.0


class TestAggregateConfidence:
    def test_uniform(self):
        from src.evaluation.confidence import aggregate_confidence
        conf = np.full((100, 100), 0.75, dtype=np.float32)
        result = aggregate_confidence(conf)
        assert np.isclose(result, 0.75)

    def test_range(self):
        from src.evaluation.confidence import aggregate_confidence
        conf = np.random.rand(64, 64).astype(np.float32)
        result = aggregate_confidence(conf)
        assert 0.0 <= result <= 1.0


class TestConfidenceMap:
    def test_compute_and_aggregate(self):
        from src.evaluation.confidence import ConfidenceMap
        cm = ConfidenceMap()
        density = np.random.rand(32, 32).astype(np.float32)
        result = cm.compute(density)
        assert result.shape == (32, 32)
        agg = cm.aggregate()
        assert 0.0 <= agg <= 1.0

    def test_threshold_mask(self):
        from src.evaluation.confidence import ConfidenceMap
        cm = ConfidenceMap()
        density = np.array([[0.1, 0.6, 0.9]], dtype=np.float32)
        cm.compute(density)
        mask = cm.threshold_mask(threshold=0.5)
        assert mask.dtype == np.uint8
        assert mask[0, 0] == 1
        assert mask[0, 2] == 0

    def test_empty_before_compute(self):
        from src.evaluation.confidence import ConfidenceMap
        cm = ConfidenceMap()
        assert cm.aggregate() == 0.0
        mask = cm.threshold_mask()
        assert isinstance(mask, np.ndarray)
