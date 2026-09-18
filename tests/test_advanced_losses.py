import pytest
import torch

from src.training.losses import (
    FilteredJaccardLoss,
    CloudConstrainedLoss,
    CombinedLoss,
)


class TestFilteredJaccardLoss:
    def test_fjl_initialization(self):
        loss_gl1 = FilteredJaccardLoss(gl_type="GL1")
        loss_gl2 = FilteredJaccardLoss(gl_type="GL2")
        assert loss_gl1.gl_type == "GL1"
        assert loss_gl2.gl_type == "GL2"

        with pytest.raises(ValueError):
            FilteredJaccardLoss(gl_type="INVALID")

    def test_fjl_identical_predictions(self):
        fjl = FilteredJaccardLoss(gl_type="GL1")
        pred = torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]], requires_grad=True)
        target = torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]])
        loss = fjl(pred, target)
        # Identical predictions on mask should yield loss near 0
        assert loss.item() < 0.05

    def test_clear_sky_behavior(self):
        """Under clear sky (target cloud S ≈ 0), LP_pc(S) ≈ 1 and HP_pc(S) ≈ 0,
        so loss is dominated by GL (Inverted Jaccard or BCE)."""
        fjl = FilteredJaccardLoss(m=1000.0, pc=0.5, p_prime_c=0.5, gl_type="GL1")
        # Target has no clouds (all 0s)
        target = torch.zeros(2, 1, 16, 16)
        pred_bad = torch.ones(2, 1, 16, 16) * 0.5  # Model falsely predicts thin haze everywhere

        loss = fjl(pred_bad, target)
        assert loss.item() > 0.0
        assert not torch.isnan(loss)

    def test_overcast_cloudy_behavior(self):
        """Under thick overcast clouds (target cloud S ≈ 1), LP_pc(S) ≈ 0 and HP_pc(S) ≈ 1,
        so loss is dominated by JL (Jaccard Loss)."""
        fjl = FilteredJaccardLoss(m=1000.0, pc=0.5, p_prime_c=0.5, gl_type="GL1")
        target = torch.ones(2, 1, 16, 16)
        pred = torch.zeros(2, 1, 16, 16)

        loss = fjl(pred, target)
        assert loss.item() > 0.9  # Maximum Jaccard distance

    def test_differentiability(self):
        fjl = FilteredJaccardLoss(gl_type="GL2")
        pred = torch.rand(4, 1, 16, 16, requires_grad=True)
        target = torch.rand(4, 1, 16, 16)

        loss = fjl(pred, target)
        loss.backward()
        assert pred.grad is not None
        assert not torch.isnan(pred.grad).any()


class TestCloudConstrainedLoss:
    def test_ccl_basic(self):
        ccl = CloudConstrainedLoss(lambda_ccl=2.0)
        b, c, h, w = 2, 3, 16, 16
        pred = torch.ones(b, c, h, w, requires_grad=True)
        target = torch.zeros(b, c, h, w)
        # Cloud mask: half cloudy (1), half clear (0)
        mask = torch.zeros(b, 1, h, w)
        mask[:, :, :8, :] = 1.0

        loss = ccl(pred, target, cloud_mask=mask)
        # L1 diff is 1 everywhere. L1_cloudy = 1.0, L1_clear = 1.0
        # Total = lambda_ccl * 1.0 + 1.0 = 2.0 * 1.0 + 1.0 = 3.0
        assert torch.isclose(loss, torch.tensor(3.0), atol=1e-4)

    def test_ccl_differentiability(self):
        ccl = CloudConstrainedLoss(lambda_ccl=1.5)
        pred = torch.rand(2, 3, 16, 16, requires_grad=True)
        target = torch.rand(2, 3, 16, 16)
        mask = (torch.rand(2, 1, 16, 16) > 0.5).float()

        loss = ccl(pred, target, cloud_mask=mask)
        loss.backward()
        assert pred.grad is not None
        assert not torch.isnan(pred.grad).any()


class TestCombinedLossIntegration:
    def test_combined_with_fjl_and_ccl(self):
        weights = {
            "l1": 1.0,
            "ssim": 1.0,
            "spectral": 0.5,
            "fjl": 1.0,
            "ccl": 1.0,
            "lambda_ccl": 1.5,
        }
        loss_fn = CombinedLoss(weights=weights)
        pred = torch.rand(2, 3, 32, 32, requires_grad=True)
        target = torch.rand(2, 3, 32, 32)
        mask = (torch.rand(2, 1, 32, 32) > 0.5).float()

        res = loss_fn(pred, target, cloud_mask=mask)
        assert "total" in res
        assert "l1" in res
        assert "ssim" in res
        assert "spectral" in res
        assert "fjl" in res
        assert "ccl" in res

        res["total"].backward()
        assert pred.grad is not None
