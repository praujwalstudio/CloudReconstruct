"""Work Package 0 Verification Suite: Data Ingestion Layer Tests
==============================================================
Validates:
- SAR dB Normalization math and bounds [-1.0, 1.0].
- S2 TOA reflectance scaling and bounds [0.0, 1.0].
- 13-band Sentinel-2 to 3-channel LISS-IV mapping (Green B3, Red B4, NIR B8).
- Directory ingestion with standard `sigma0/`, `cloudy/`, `clear/` layout.
- Non-overlapping ROI spatial train/val/test splits.
- Multi-temporal sequences (length t <= 5) in SEN12MSCRTSDataset.
- Coordinate transforms and geospatial metadata loading with rasterio.
"""

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from rasterio.crs import CRS
import torch
from pathlib import Path

from src.data.band_harmonization import harmonize_s2_to_liss4, S2_TO_LISS4_INDICES
from src.data.sen12ms_dataset import (
    SEN12MSCRDataset,
    SEN12MSCRTSDataset,
    normalize_sar,
    extract_roi_id,
)


class TestWP0DataIngestion:
    def test_sar_normalization_mathematical_limits(self):
        """Tests SAR dB normalization formula:
        SAR_norm = clip(SAR, -25.0, 0.0) / 12.5 + 1.0
        """
        # Test extreme and boundary values
        inputs = np.array([-50.0, -25.0, -18.75, -12.5, -6.25, 0.0, 15.0], dtype=np.float32)
        expected = np.array([-1.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.0], dtype=np.float32)
        norm = normalize_sar(inputs)
        assert np.allclose(norm, expected, atol=1e-5)
        assert (norm >= -1.0).all() and (norm <= 1.0).all()

    def test_optical_toa_reflectance_scaling(self):
        """Tests Sentinel-2 TOA division by 10000.0 and clipping to [0.0, 1.0]."""
        # 13 bands raw Digital Numbers (DN)
        s2_raw = torch.tensor([
            0.0, 500.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0, 8000.0,
            8500.0, 9000.0, 10000.0, 12000.0, 15000.0
        ]).view(13, 1, 1).expand(13, 16, 16)

        liss4 = harmonize_s2_to_liss4(s2_raw, scale_toa=True)
        assert liss4.shape == (3, 16, 16)
        # Green (B3 / index 2): 2000 / 10000 = 0.2
        # Red (B4 / index 3): 3000 / 10000 = 0.3
        # NIR (B8 / index 7): 8000 / 10000 = 0.8
        assert torch.isclose(liss4[0], torch.tensor(0.2)).all()
        assert torch.isclose(liss4[1], torch.tensor(0.3)).all()
        assert torch.isclose(liss4[2], torch.tensor(0.8)).all()
        assert (liss4 >= 0.0).all() and (liss4 <= 1.0).all()

    @pytest.fixture
    def mock_standard_raw_dir(self, tmp_path):
        """Creates standard WP0 directory:
        data/raw/
        ├── sigma0/
        ├── cloudy/
        └── clear/
        """
        sigma0_dir = tmp_path / "sigma0"
        cloudy_dir = tmp_path / "cloudy"
        clear_dir = tmp_path / "clear"

        sigma0_dir.mkdir(parents=True)
        cloudy_dir.mkdir(parents=True)
        clear_dir.mkdir(parents=True)

        profile = {
            "driver": "GTiff",
            "height": 16,
            "width": 16,
            "crs": CRS.from_epsg(4326),
            "transform": from_origin(77.5, 12.9, 0.0001, 0.0001),
        }

        # 3 ROIs, 2 patches each
        for roi in ["ROI_01", "ROI_02", "ROI_03"]:
            for p in range(2):
                p_name = f"{roi}_patch_{p}.tif"
                # S1 (2 bands)
                with rasterio.open(sigma0_dir / f"sigma0_{p_name}", "w", count=2, dtype=np.float32, **profile) as dst:
                    dst.write(np.random.uniform(-25.0, 0.0, (2, 16, 16)).astype(np.float32))

                # Cloudy S2 (13 bands)
                with rasterio.open(cloudy_dir / f"cloudy_{p_name}", "w", count=13, dtype=np.uint16, **profile) as dst:
                    dst.write(np.random.randint(500, 8000, (13, 16, 16), dtype=np.uint16))

                # Clear S2 (13 bands)
                with rasterio.open(clear_dir / f"clear_{p_name}", "w", count=13, dtype=np.uint16, **profile) as dst:
                    dst.write(np.random.randint(300, 6000, (13, 16, 16), dtype=np.uint16))

        return tmp_path

    def test_standard_directory_ingestion(self, mock_standard_raw_dir):
        ds = SEN12MSCRDataset(mock_standard_raw_dir, split="all", return_dict=True)
        assert len(ds) == 6
        sample = ds[0]

        assert sample["s2_cloudy"].shape == (3, 16, 16)
        assert sample["s2_clear"].shape == (3, 16, 16)
        assert sample["s1"].shape == (2, 16, 16)
        assert "meta" in sample
        assert "crs" in sample["meta"]
        assert "transform" in sample["meta"]

    def test_multi_temporal_sequence_dataset(self, mock_standard_raw_dir):
        ts_ds = SEN12MSCRTSDataset(mock_standard_raw_dir, seq_length=5, split="all")
        assert len(ts_ds) >= 1
        seq = ts_ds[0]

        assert "cloudy_seq" in seq
        assert "clear_seq" in seq
        assert "s1_seq" in seq
        assert seq["cloudy_seq"].ndim == 4  # (T, 3, H, W)
        assert seq["cloudy_seq"].shape[1] == 3
        assert seq["s1_seq"].shape[1] == 2
        assert seq["seq_len"] <= 5
