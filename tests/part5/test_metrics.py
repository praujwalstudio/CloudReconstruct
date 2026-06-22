import numpy as np


class TestPSNR:
    def test_identical_images(self):
        from src.evaluation.metrics import psnr
        img = np.random.randint(0, 1023, (32, 32, 3), dtype=np.uint16)
        assert psnr(img, img) == float("inf")

    def test_different_images(self):
        from src.evaluation.metrics import psnr
        a = np.ones((16, 16, 3), dtype=np.uint16) * 500
        b = np.ones((16, 16, 3), dtype=np.uint16) * 600
        result = psnr(a, b, data_range=1023)
        assert result > 0
        assert result < 100

    def test_with_mask(self):
        from src.evaluation.metrics import psnr
        a = np.ones((16, 16, 3), dtype=np.uint16) * 500
        b = np.ones((16, 16, 3), dtype=np.uint16) * 600
        mask = np.ones((16, 16), dtype=np.uint8)
        result = psnr(a, b, data_range=1023, mask=mask)
        assert result > 0

    def test_shape_mismatch_raises(self):
        from src.evaluation.metrics import psnr
        import pytest
        with pytest.raises(ValueError, match="Shape mismatch"):
            psnr(np.ones((10, 10, 3)), np.ones((20, 20, 3)))

    def test_auto_data_range_uint16(self):
        from src.evaluation.metrics import psnr
        a = np.zeros((8, 8, 3), dtype=np.uint16)
        b = np.ones((8, 8, 3), dtype=np.uint16) * 100
        result = psnr(a, b)
        assert result > 0


class TestSAM:
    def test_identical_images(self):
        from src.evaluation.metrics import sam
        img = np.random.rand(16, 16, 3).astype(np.float32)
        assert sam(img, img) < 0.5

    def test_opposite_spectra(self):
        from src.evaluation.metrics import sam
        a = np.array([[[1, 0, 0]]], dtype=np.float32)
        b = np.array([[[0, 1, 0]]], dtype=np.float32)
        angle = sam(a, b)
        assert angle > 80
        assert angle < 100

    def test_shape_mismatch_raises(self):
        from src.evaluation.metrics import sam
        import pytest
        with pytest.raises(ValueError, match="Shape mismatch"):
            sam(np.ones((10, 10, 3)), np.ones((20, 20, 3)))

    def test_with_mask(self):
        from src.evaluation.metrics import sam
        a = np.random.rand(16, 16, 3).astype(np.float32)
        b = np.random.rand(16, 16, 3).astype(np.float32)
        mask = np.ones((16, 16), dtype=np.uint8)
        result = sam(a, b, mask=mask)
        assert result >= 0


class TestNDVICorrelation:
    def test_perfect_correlation(self):
        from src.evaluation.metrics import ndvi_correlation
        img = np.random.randint(0, 1023, (32, 32, 3), dtype=np.uint16)
        corr = ndvi_correlation(img, img)
        assert abs(corr - 1.0) < 1e-6

    def test_uncorrelated(self):
        from src.evaluation.metrics import ndvi_correlation
        rng = np.random.default_rng(42)
        a = rng.integers(0, 1023, (16, 16, 3), dtype=np.uint16)
        b = a.copy()
        b[..., 1] = a[..., 2].copy()  # swap RED ↔ NIR → NDVI flips sign
        b[..., 2] = a[..., 1].copy()
        corr = ndvi_correlation(a, b)
        assert corr < -0.99
        assert corr >= -1.0

    def test_output_range(self):
        from src.evaluation.metrics import ndvi_correlation
        a = np.random.randint(0, 1023, (20, 20, 3), dtype=np.uint16)
        b = np.random.randint(0, 1023, (20, 20, 3), dtype=np.uint16)
        corr = ndvi_correlation(a, b)
        assert -1.0 <= corr <= 1.0

    def test_with_mask(self):
        from src.evaluation.metrics import ndvi_correlation
        a = np.random.randint(0, 1023, (20, 20, 3), dtype=np.uint16)
        b = np.random.randint(0, 1023, (20, 20, 3), dtype=np.uint16)
        mask = np.ones((20, 20), dtype=np.uint8)
        mask[:5] = 0
        corr = ndvi_correlation(a, b, mask=mask)
        assert isinstance(corr, float)


class TestComputeAllMetrics:
    def test_returns_dict_with_keys(self):
        from src.evaluation.metrics import compute_all_metrics
        a = np.random.randint(0, 1023, (16, 16, 3), dtype=np.uint16)
        b = np.random.randint(0, 1023, (16, 16, 3), dtype=np.uint16)
        result = compute_all_metrics(a, b)
        assert isinstance(result, dict)
        assert "psnr" in result
        assert "sam" in result
        assert "ndvi_correlation" in result

    def test_with_mask(self):
        from src.evaluation.metrics import compute_all_metrics
        a = np.random.randint(0, 1023, (16, 16, 3), dtype=np.uint16)
        b = np.random.randint(0, 1023, (16, 16, 3), dtype=np.uint16)
        mask = np.ones((16, 16), dtype=np.uint8)
        result = compute_all_metrics(a, b, mask=mask)
        assert result["psnr"] > 0
