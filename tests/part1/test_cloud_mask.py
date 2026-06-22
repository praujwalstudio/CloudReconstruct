import numpy as np
import pytest
from src.preprocessing.cloud_mask import (
    compute_cloud_mask_ndvi,
    compute_cloud_mask_brightness,
    compute_cloud_mask_whiteness,
    compute_cloud_mask_temporal,
    ensemble_masks,
    refine_mask,
    compute_cloud_mask,
    cloud_density,
    classify_cloud_density,
)


class TestNDVIMask:
    def test_high_ndvi_no_cloud(self):
        image = np.zeros((64, 64, 3), dtype=np.float32)
        image[..., 1] = 0.1  # red (low)
        image[..., 2] = 0.5  # nir (high) → NDVI ≈ 0.67 → not cloud
        mask = compute_cloud_mask_ndvi(image, threshold=0.2)
        assert mask.dtype == np.uint8
        assert mask.shape == (64, 64)
        assert mask.mean() < 0.5

    def test_low_ndvi_detects_cloud(self):
        image = np.zeros((64, 64, 3), dtype=np.float32)
        image[..., 1] = 0.5  # red (high)
        image[..., 2] = 0.3  # nir (lower) → NDVI < 0 → likely cloud
        mask = compute_cloud_mask_ndvi(image, threshold=0.2)
        assert mask.mean() > 0.5


class TestBrightnessMask:
    def test_bright_pixels_detected(self):
        image = np.ones((64, 64, 3), dtype=np.float32) * 0.9  # very bright
        # With percentile=50, threshold is 0.9 which equals all pixels.
        # Use percentile=99 so threshold is 0.9 and all 0.9-pixels are above threshold.
        # Actually: all pixels are 0.9, mean=0.9, percentile=99 gives 0.9.
        # We need some pixels above the threshold. Make half bright, half dark.
        image[:32, :, :] = 0.9  # bright top half
        image[32:, :, :] = 0.1  # dark bottom half
        mask = compute_cloud_mask_brightness(image, percentile=50)
        assert mask.dtype == np.uint8
        assert mask.mean() > 0.4  # ~half the pixels should be detected

    def test_dark_pixels_not_detected(self):
        image = np.ones((64, 64, 3), dtype=np.float32) * 0.1  # dark
        mask = compute_cloud_mask_brightness(image, percentile=90)
        assert mask.mean() == 0


class TestWhitenessMask:
    def test_white_pixels_detected(self):
        image = np.ones((64, 64, 3), dtype=np.float32) * 0.5  # uniform = white
        mask = compute_cloud_mask_whiteness(image, threshold=0.05)
        assert mask.dtype == np.uint8

    def test_colored_pixels_not_white(self):
        image = np.zeros((64, 64, 3), dtype=np.float32)
        image[..., 0] = 0.1
        image[..., 1] = 0.5
        image[..., 2] = 0.3
        mask = compute_cloud_mask_whiteness(image)
        assert mask.sum() < image.size / 2


class TestTemporalMask:
    def test_large_diff_detects_cloud(self):
        ref = np.ones((64, 64, 3), dtype=np.float32) * 0.3
        img = np.ones((64, 64, 3), dtype=np.float32) * 0.8  # very different
        mask = compute_cloud_mask_temporal(img, ref, threshold=0.1)
        assert mask.dtype == np.uint8
        assert mask.mean() > 0.5


class TestEnsemble:
    def test_majority_vote(self):
        # 3 masks: 2 say cloud, 1 says clear → should be cloud
        masks = [
            np.ones((16, 16), dtype=np.uint8),
            np.ones((16, 16), dtype=np.uint8),
            np.zeros((16, 16), dtype=np.uint8),
        ]
        result = ensemble_masks(masks, method="majority")
        assert result.mean() == 1.0

    def test_union(self):
        masks = [
            np.zeros((16, 16), dtype=np.uint8),
            np.ones((16, 16), dtype=np.uint8),
        ]
        result = ensemble_masks(masks, method="union")
        assert result.mean() == 1.0

    def test_intersection(self):
        masks = [
            np.ones((16, 16), dtype=np.uint8),
            np.zeros((16, 16), dtype=np.uint8),
        ]
        result = ensemble_masks(masks, method="intersection")
        assert result.mean() == 0.0


class TestRefineMask:
    def test_morphological_cleanup(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[30:34, 30:34] = 1  # small cluster
        refined = refine_mask(mask, kernel_size=5)
        assert refined.shape == mask.shape
        assert refined.dtype == np.uint8


class TestComputeCloudMask:
    def test_default_params_return_mask(self, sample_cloudy_image):
        mask = compute_cloud_mask(sample_cloudy_image)
        assert mask.dtype == np.uint8
        assert mask.shape[:2] == sample_cloudy_image.shape[:2]

    def test_with_temporal_reference(self, sample_cloudy_image, sample_clear_image):
        mask = compute_cloud_mask(sample_cloudy_image, reference=sample_clear_image)
        assert mask.shape[:2] == sample_cloudy_image.shape[:2]

    def test_disabled_all_raises(self):
        with pytest.raises(ValueError):
            compute_cloud_mask(
                np.ones((16, 16, 3), dtype=np.float32),
                use_ndvi=False,
                use_brightness=False,
                use_whiteness=False,
            )


class TestCloudDensity:
    def test_output_shape_and_range(self, sample_binary_mask):
        density = cloud_density(sample_binary_mask, patch_size=32)
        assert density.shape == sample_binary_mask.shape
        assert density.dtype == np.float32
        assert 0 <= density.min() <= density.max() <= 1


class TestClassifyCloudDensity:
    def test_output_values(self):
        density = np.array([
            [0.1, 0.4],
            [0.6, 0.9],
        ])
        classes = classify_cloud_density(density)
        assert classes[0, 0] == 0  # clear
        assert classes[0, 1] == 1  # thin
        assert classes[1, 0] == 2  # medium
        assert classes[1, 1] == 3  # dense
