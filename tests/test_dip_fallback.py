import pytest
import torch
import numpy as np
from src.evaluation.dip_fallback import DIPNet, DIPInpainter, total_variation_loss
from src.evaluation.confidence import compute_confidence, ConfidenceMap, generate_fallback_mask
from src.evaluation.analysis_readiness import compute_ars, AnalysisReadiness


class TestDIPFallback:
    def test_dip_net_shape(self):
        net = DIPNet(in_channels=8, out_channels=3, hidden_dims=[8, 16, 32])
        z = torch.randn(1, 8, 32, 32)
        out = net(z)
        assert out.shape == (1, 3, 32, 32)
        assert (out >= 0.0).all() and (out <= 1.0).all()

    def test_tv_loss(self):
        flat = torch.ones(1, 3, 32, 32)
        tv_flat = total_variation_loss(flat)
        assert tv_flat.item() == 0.0

        noisy = torch.randn(1, 3, 32, 32)
        tv_noisy = total_variation_loss(noisy)
        assert tv_noisy.item() > 0.0

    def test_dip_inpainting_convergence(self):
        inpainter = DIPInpainter(z_channels=4, out_channels=3)
        cloudy = torch.rand(1, 3, 32, 32)
        mask = torch.zeros(1, 1, 32, 32)
        mask[:, :, 10:22, 10:22] = 1.0  # Center cloud occlusion

        inpainted = inpainter.inpaint(cloudy, mask, num_iters=15, lr=0.05)
        assert inpainted.shape == (1, 3, 32, 32)
        assert torch.isfinite(inpainted).all()
        assert (inpainted >= 0.0).all() and (inpainted <= 1.0).all()

    def test_blend_fallback(self):
        rec = torch.zeros(1, 3, 16, 16)
        dip = torch.ones(1, 3, 16, 16)
        fallback_mask = torch.zeros(1, 1, 16, 16)
        fallback_mask[:, :, 4:12, 4:12] = 1.0

        blended = DIPInpainter.blend_fallback(rec, dip, fallback_mask)
        assert blended.shape == (1, 3, 16, 16)
        assert (blended[:, :, 4:12, 4:12] == 1.0).all()
        assert (blended[:, :, 0:4, 0:4] == 0.0).all()


class TestConfidenceCalibration:
    def test_confidence_with_calibrated_variance(self):
        density = np.zeros((32, 32), dtype=np.float32)
        low_var = np.full((32, 32), 0.01, dtype=np.float32)
        high_var = np.full((32, 32), 2.0, dtype=np.float32)

        conf_low_var = compute_confidence(density, calibrated_variance=low_var)
        conf_high_var = compute_confidence(density, calibrated_variance=high_var)

        assert float(np.mean(conf_low_var)) > float(np.mean(conf_high_var))
        assert (conf_low_var >= 0.0).all() and (conf_low_var <= 1.0).all()

    def test_fallback_mask_generation(self):
        conf = np.ones((32, 32), dtype=np.float32)
        conf[10:20, 10:20] = 0.2  # Low confidence pocket

        mask = generate_fallback_mask(conf, threshold=0.4)
        assert mask.shape == (32, 32)
        assert mask[15, 15] == 1
        assert mask[0, 0] == 0

    def test_confidence_map_fallback(self):
        cmap = ConfidenceMap()
        density = np.zeros((16, 16), dtype=np.float32)
        density[4:12, 4:12] = 0.9
        cmap.compute(density)

        fallback = cmap.fallback_mask(threshold=0.3)
        assert fallback.shape == (16, 16)
        assert fallback[8, 8] == 1


class TestARSWithCalibration:
    def test_compute_ars(self):
        conf = np.full((32, 32), 0.95, dtype=np.float32)
        img_corr = np.random.rand(32, 32, 3).astype(np.float32)
        img_ref = img_corr.copy()

        ars_dict = compute_ars(conf, img_corr, img_ref)
        assert "ars" in ars_dict
        assert ars_dict["ars"] >= 0.9
        ar = AnalysisReadiness()
        assert ar.grade(ars_dict["ars"]) == "A"
