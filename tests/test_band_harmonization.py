import numpy as np
import pytest
import torch

from src.data.band_harmonization import harmonize_s2_to_liss4, S2_TO_LISS4_INDICES


class TestBandHarmonization:
    def test_band_indices(self):
        assert S2_TO_LISS4_INDICES == [2, 3, 7]  # B3 (Green), B4 (Red), B8 (NIR)

    def test_tensor_4d(self):
        # Batch of 13-band S2 tensors
        b, c, h, w = 2, 13, 32, 32
        s2 = torch.ones(b, c, h, w, dtype=torch.float32) * 5000.0
        # Set distinct values for Green, Red, NIR
        s2[:, 2, :, :] = 2000.0  # Green
        s2[:, 3, :, :] = 3000.0  # Red
        s2[:, 7, :, :] = 8000.0  # NIR

        liss4 = harmonize_s2_to_liss4(s2, scale_toa=True)
        assert liss4.shape == (2, 3, 32, 32)
        assert torch.isclose(liss4[:, 0, :, :], torch.tensor(0.2)).all()  # 2000 / 10000
        assert torch.isclose(liss4[:, 1, :, :], torch.tensor(0.3)).all()  # 3000 / 10000
        assert torch.isclose(liss4[:, 2, :, :], torch.tensor(0.8)).all()  # 8000 / 10000

    def test_tensor_3d(self):
        c, h, w = 13, 16, 16
        s2 = torch.full((c, h, w), 4000.0)
        liss4 = harmonize_s2_to_liss4(s2, scale_toa=True)
        assert liss4.shape == (3, 16, 16)
        assert torch.isclose(liss4, torch.tensor(0.4)).all()

    def test_numpy_channel_last(self):
        h, w, c = 20, 20, 13
        s2 = np.full((h, w, c), 6000.0, dtype=np.float32)
        s2[:, :, 2] = 1000.0
        s2[:, :, 3] = 4000.0
        s2[:, :, 7] = 7000.0

        liss4 = harmonize_s2_to_liss4(s2, scale_toa=True)
        assert liss4.shape == (20, 20, 3)
        assert np.isclose(liss4[:, :, 0], 0.1).all()
        assert np.isclose(liss4[:, :, 1], 0.4).all()
        assert np.isclose(liss4[:, :, 2], 0.7).all()

    def test_numpy_channel_first(self):
        c, h, w = 13, 20, 20
        s2 = np.full((c, h, w), 2500.0, dtype=np.float32)
        liss4 = harmonize_s2_to_liss4(s2, scale_toa=True)
        assert liss4.shape == (3, 20, 20)
        assert np.isclose(liss4, 0.25).all()

    def test_clipping(self):
        # Values exceeding TOA scale are clipped to 1.0
        s2 = torch.full((13, 10, 10), 15000.0)
        liss4 = harmonize_s2_to_liss4(s2, scale_toa=True, clip_range=(0.0, 1.0))
        assert (liss4 <= 1.0).all()
        assert (liss4 >= 0.0).all()

    def test_invalid_shape_raises(self):
        with pytest.raises(ValueError):
            harmonize_s2_to_liss4(torch.ones(2, 5, 10, 10))
