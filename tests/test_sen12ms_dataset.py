import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
import torch
from pathlib import Path

from src.data.sen12ms_dataset import (
    SEN12MSCRDataset,
    normalize_sar,
    extract_roi_id,
)


class TestSEN12MSDataset:
    def test_normalize_sar(self):
        sar = np.array([-30.0, -25.0, -12.5, 0.0, 10.0], dtype=np.float32)
        norm = normalize_sar(sar)
        # -30 clips to -25 -> -25/12.5 + 1 = -1.0
        # -25 -> -1.0
        # -12.5 -> -12.5/12.5 + 1 = 0.0
        # 0.0 -> 0/12.5 + 1 = 1.0
        # 10.0 clips to 0.0 -> 1.0
        expected = np.array([-1.0, -1.0, 0.0, 1.0, 1.0], dtype=np.float32)
        assert np.allclose(norm, expected, atol=1e-5)
        assert norm.min() >= -1.0
        assert norm.max() <= 1.0

    def test_extract_roi_id(self):
        assert extract_roi_id("ROIs1158_spring_s2_cloudy_patch_42.tif") == "ROIs1158_spring"
        assert extract_roi_id("data/ROIs1868_summer/s1/patch_10.tif") == "ROIs1868_summer"
        assert extract_roi_id("scene_ROI003_patch_15.tif") == "scene_ROI003"

    @pytest.fixture
    def mock_sen12ms_dir(self, tmp_path):
        """Creates a mock SEN12MS dataset hierarchy with 4 ROIs."""
        rois = ["ROIs1158_spring", "ROIs1868_summer", "ROIs1970_fall", "ROIs2017_winter"]
        profile = {
            "driver": "GTiff",
            "height": 32,
            "width": 32,
            "crs": None,
            "transform": from_origin(0, 0, 1, 1),
        }

        for roi in rois:
            roi_dir = tmp_path / roi
            roi_dir.mkdir(parents=True, exist_ok=True)
            for p_idx in range(3):
                # 1. Cloudy S2 (13 bands)
                p_s2_cloudy = roi_dir / f"{roi}_s2_cloudy_patch_{p_idx}.tif"
                with rasterio.open(p_s2_cloudy, "w", count=13, dtype=np.uint16, **profile) as dst:
                    data = np.random.randint(500, 4000, (13, 32, 32), dtype=np.uint16)
                    dst.write(data)

                # 2. Clear S2 (13 bands)
                p_s2_clear = roi_dir / f"{roi}_s2_clear_patch_{p_idx}.tif"
                with rasterio.open(p_s2_clear, "w", count=13, dtype=np.uint16, **profile) as dst:
                    data = np.random.randint(300, 2500, (13, 32, 32), dtype=np.uint16)
                    dst.write(data)

                # 3. SAR S1 (2 bands: VV, VH in dB)
                p_s1 = roi_dir / f"{roi}_s1_patch_{p_idx}.tif"
                with rasterio.open(p_s1, "w", count=2, dtype=np.float32, **profile) as dst:
                    data = np.random.uniform(-25.0, -5.0, (2, 32, 32)).astype(np.float32)
                    dst.write(data)

                # 4. DEM (1 band)
                p_dem = roi_dir / f"{roi}_dem_patch_{p_idx}.tif"
                with rasterio.open(p_dem, "w", count=1, dtype=np.float32, **profile) as dst:
                    data = np.random.uniform(100.0, 800.0, (1, 32, 32)).astype(np.float32)
                    dst.write(data)

        return tmp_path

    def test_dataset_loading_and_shapes(self, mock_sen12ms_dir):
        ds = SEN12MSCRDataset(mock_sen12ms_dir, split="all", return_dict=True)
        assert len(ds) == 12  # 4 ROIs * 3 patches
        sample = ds[0]

        assert "s2_cloudy" in sample
        assert "s2_clear" in sample
        assert "s1" in sample
        assert "dem" in sample
        assert "roi_id" in sample

        # Check harmonized LISS-IV 3-channel shape
        assert sample["s2_cloudy"].shape == (3, 32, 32)
        assert sample["s2_clear"].shape == (3, 32, 32)
        # Check SAR 2-channel shape
        assert sample["s1"].shape == (2, 32, 32)

        # Check value ranges
        assert sample["s2_cloudy"].min() >= 0.0 and sample["s2_cloudy"].max() <= 1.0
        assert sample["s1"].min() >= -1.0 and sample["s1"].max() <= 1.0

    def test_roi_split_isolation(self, mock_sen12ms_dir):
        """Verifies zero ROI overlap across train, val, and test splits (no spatial leakage)."""
        train_ds = SEN12MSCRDataset(mock_sen12ms_dir, split="train", train_ratio=0.5, val_ratio=0.25, seed=42)
        val_ds = SEN12MSCRDataset(mock_sen12ms_dir, split="val", train_ratio=0.5, val_ratio=0.25, seed=42)
        test_ds = SEN12MSCRDataset(mock_sen12ms_dir, split="test", train_ratio=0.5, val_ratio=0.25, seed=42)

        train_rois = set(s["roi_id"] for s in train_ds.samples)
        val_rois = set(s["roi_id"] for s in val_ds.samples)
        test_rois = set(s["roi_id"] for s in test_ds.samples)

        # Must be completely disjoint sets
        assert train_rois.isdisjoint(val_rois)
        assert train_rois.isdisjoint(test_rois)
        assert val_rois.isdisjoint(test_rois)
        assert len(train_rois | val_rois | test_rois) == 4
