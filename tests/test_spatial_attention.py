import pytest
import torch
import torch.nn as nn
from src.models.spatial_attention import SpatialAttentionBlock, SpatialAttentionGenerator
from src.models.discriminator import PatchGANDiscriminator
from src.models.thin_cloud_correction import ThinCloudCorrection
from src.training.losses import SpatialAttentionLoss, LSGANLoss, CARLLoss
from src.training.train_correction import CorrectionTrainer


class TestSpatialAttention:
    def test_sab_block_forward(self):
        sab = SpatialAttentionBlock(channels=32)
        x = torch.randn(2, 32, 64, 64)
        out, attn = sab(x)
        assert out.shape == (2, 32, 64, 64)
        assert attn.shape == (2, 1, 64, 64)
        assert (attn >= 0.0).all() and (attn <= 1.0).all()

    def test_generator_forward(self):
        gen = SpatialAttentionGenerator(in_channels=3, density_channels=1, out_channels=3, num_features=32, num_blocks=3)
        cloudy = torch.rand(2, 3, 64, 64)
        density = torch.rand(2, 1, 64, 64)

        out = gen(cloudy, density)
        assert out.shape == (2, 3, 64, 64)
        assert (out >= 0.0).all() and (out <= 1.0).all()

        out_with_attn, attn = gen(cloudy, density, return_attn=True)
        assert out_with_attn.shape == (2, 3, 64, 64)
        assert attn.shape == (2, 1, 64, 64)
        assert (attn >= 0.0).all() and (attn <= 1.0).all()

    def test_thin_cloud_correction_with_spa(self):
        model = ThinCloudCorrection(in_channels=3, use_spatial_attention=True, num_blocks=2)
        cloudy = torch.rand(1, 3, 32, 32)
        density = torch.rand(1, 1, 32, 32)
        out = model(cloudy, density)
        assert out.shape == (1, 3, 32, 32)


class TestDiscriminator:
    def test_patchgan_output_shape(self):
        disc = PatchGANDiscriminator(in_channels=3, num_filters=32, num_layers=3)
        x = torch.randn(2, 3, 64, 64)
        out = disc(x)
        assert out.ndim == 4
        assert out.shape[0] == 2
        assert out.shape[1] == 1
        # Receptive field downsamples spatial dimensions
        assert out.shape[2] < 64 and out.shape[3] < 64

    def test_conditional_patchgan(self):
        disc = PatchGANDiscriminator(in_channels=3, condition_channels=1, num_filters=32)
        x = torch.randn(2, 3, 64, 64)
        cond = torch.randn(2, 1, 64, 64)
        out = disc(x, condition=cond)
        assert out.shape[0] == 2
        assert out.shape[1] == 1


class TestAdversarialLosses:
    def test_spatial_attention_loss(self):
        loss_fn = SpatialAttentionLoss()
        pred = torch.ones(2, 3, 32, 32)
        target = torch.zeros(2, 3, 32, 32)
        attn = torch.ones(2, 1, 32, 32) * 0.5
        loss = loss_fn(pred, target, attn)
        assert torch.isfinite(loss)
        assert torch.allclose(loss, torch.tensor(1.0), atol=1e-3)

    def test_lsgan_loss(self):
        loss_fn = LSGANLoss()
        d_real = torch.ones(2, 1, 8, 8)
        d_fake = torch.zeros(2, 1, 8, 8)
        loss_g = loss_fn.generator_loss(d_fake)
        loss_d = loss_fn.discriminator_loss(d_real, d_fake)
        assert torch.isfinite(loss_g)
        assert torch.isfinite(loss_d)
        assert torch.allclose(loss_d, torch.tensor(0.0), atol=1e-4)

    def test_carl_loss(self):
        carl = CARLLoss(lambda_l1=1.0, lambda_ssim=0.5, lambda_att=0.5, lambda_adv=0.01)
        pred = torch.rand(2, 3, 32, 32)
        target = torch.rand(2, 3, 32, 32)
        attn = torch.rand(2, 1, 32, 32)
        d_fake = torch.rand(2, 1, 8, 8)
        losses = carl(pred, target, attention_map=attn, d_fake=d_fake)
        assert "total" in losses
        assert "l1" in losses
        assert "ssim" in losses
        assert "att" in losses
        assert "adv" in losses
        assert torch.isfinite(losses["total"])


class TestAdversarialTrainer:
    def test_train_epoch_with_discriminator(self):
        from torch.utils.data import TensorDataset, DataLoader

        gen = SpatialAttentionGenerator(in_channels=3, density_channels=1, out_channels=3, num_features=16, num_blocks=2)
        disc = PatchGANDiscriminator(in_channels=3, num_filters=16, num_layers=2)

        trainer = CorrectionTrainer(model=gen, discriminator=disc)

        cloudy = torch.rand(4, 3, 32, 32)
        density = torch.rand(4, 1, 32, 32)
        clear = torch.rand(4, 3, 32, 32)

        ds = TensorDataset(cloudy, density, clear)
        loader = DataLoader(ds, batch_size=2)

        metrics = trainer.train_epoch(loader)
        assert "total" in metrics
        assert "loss_d" in metrics
        assert metrics["total"] >= 0.0
