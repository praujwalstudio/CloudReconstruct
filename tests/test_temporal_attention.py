import pytest
import torch
import torch.nn as nn
from src.models.temporal_fusion import (
    CrossTemporalAttention2d,
    UncertaintyHead,
    TemporalFusion,
    MultiTemporalFusion,
)
from src.training.losses import HeteroscedasticNLLLoss
from src.training.train_temporal import TemporalTrainer


class TestCrossTemporalAttention:
    def test_forward_shape(self):
        attn = CrossTemporalAttention2d(in_dim=32, num_heads=4)
        target = torch.randn(2, 32, 64, 64)
        ref = torch.randn(2, 32, 64, 64)
        out = attn(target, ref)
        assert out.shape == (2, 32, 64, 64)
        assert torch.isfinite(out).all()

    def test_gradient_flow(self):
        attn = CrossTemporalAttention2d(in_dim=32, num_heads=2)
        target = torch.randn(2, 32, 32, 32, requires_grad=True)
        ref = torch.randn(2, 32, 32, 32, requires_grad=True)
        out = attn(target, ref)
        loss = out.sum()
        loss.backward()
        assert target.grad is not None and ref.grad is not None


class TestUncertaintyHead:
    def test_log_variance_bounds(self):
        head = UncertaintyHead(in_features=64, out_channels=3)
        feats = torch.randn(2, 64, 32, 32)
        log_var = head(feats)
        assert log_var.shape == (2, 3, 32, 32)
        assert (log_var >= -10.0).all() and (log_var <= 5.0).all()


class TestHeteroscedasticNLLLoss:
    def test_loss_computation(self):
        loss_fn = HeteroscedasticNLLLoss()
        pred = torch.ones(2, 3, 32, 32)
        target = torch.ones(2, 3, 32, 32)
        log_var = torch.zeros(2, 3, 32, 32)  # var = 1.0

        # Exact match with s=0 -> 0.5 * (1 * 0 + 0) = 0.0
        loss = loss_fn(pred, target, log_var)
        assert torch.allclose(loss, torch.tensor(0.0), atol=1e-5)

    def test_loss_gradient(self):
        loss_fn = HeteroscedasticNLLLoss()
        pred = torch.rand(2, 3, 32, 32, requires_grad=True)
        target = torch.rand(2, 3, 32, 32)
        log_var = torch.zeros(2, 3, 32, 32, requires_grad=True)

        loss = loss_fn(pred, target, log_var)
        loss.backward()
        assert pred.grad is not None
        assert log_var.grad is not None


class TestTemporalFusionWithUncertainty:
    def test_forward_with_uncertainty(self):
        model = TemporalFusion(in_channels=3, hidden=32)
        cloudy = torch.rand(2, 3, 32, 32)
        ref = torch.rand(2, 3, 32, 32)
        density = torch.rand(2, 1, 32, 32)

        out, log_var = model(cloudy, ref, density, return_uncertainty=True)
        assert out.shape == (2, 3, 32, 32)
        assert log_var.shape == (2, 3, 32, 32)
        assert (out >= 0.0).all() and (out <= 1.0).all()

    def test_multi_temporal_uncertainty(self):
        model = MultiTemporalFusion(in_channels=3, hidden=32)
        cloudy = torch.rand(2, 3, 32, 32)
        refs = [torch.rand(2, 3, 32, 32) for _ in range(3)]
        density = torch.rand(2, 1, 32, 32)

        out, composite_log_var, fused_list = model(cloudy, refs, density, return_uncertainty=True)
        assert out.shape == (2, 3, 32, 32)
        assert composite_log_var.shape == (2, 3, 32, 32)
        assert len(fused_list) == 3


class TestTemporalTrainerWithUncertainty:
    def test_trainer_epoch(self):
        from torch.utils.data import TensorDataset, DataLoader

        model = TemporalFusion(in_channels=3, hidden=16)
        trainer = TemporalTrainer(model, use_uncertainty=True)

        cloudy = torch.rand(4, 3, 32, 32)
        ref = torch.rand(4, 3, 32, 32)
        target = torch.rand(4, 3, 32, 32)
        density = torch.rand(4, 1, 32, 32)

        ds = TensorDataset(cloudy, ref, target, density)
        loader = DataLoader(ds, batch_size=2)

        metrics = trainer.train_epoch(loader)
        assert "total" in metrics
        assert "nll" in metrics
        assert torch.isfinite(torch.tensor(metrics["total"]))
