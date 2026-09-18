import pytest
import torch
import torch.nn as nn
from src.models.ir_sde import IRSDE
from src.models.sar_fusion import SARDiffusionWrapper, SARConditionalUNet


class TestIRSDE:
    def test_schedules(self):
        sde_cos = IRSDE(num_timesteps=50, schedule="cosine")
        sde_exp = IRSDE(num_timesteps=50, schedule="exponential")
        sde_lin = IRSDE(num_timesteps=50, schedule="linear")

        t = torch.tensor([0.0, 0.5, 1.0])
        mu_cos, sig_cos = sde_cos.get_schedule(t)
        assert mu_cos[0] > mu_cos[1] > mu_cos[2]
        assert sig_cos.shape == (3,)

        mu_exp, sig_exp = sde_exp.get_schedule(t)
        assert mu_exp[0] > mu_exp[2]

    def test_q_sample_mean_reversion(self):
        sde = IRSDE(num_timesteps=100, schedule="cosine")
        x_0 = torch.zeros(2, 3, 32, 32)
        y = torch.ones(2, 3, 32, 32)

        # At t=0, x_t should be almost purely x_0
        t_0 = torch.tensor([0.0, 0.0])
        noise_zero = torch.zeros_like(x_0)
        x_t0, _ = sde.q_sample(x_0, y, t_0, noise=noise_zero)
        assert torch.allclose(x_t0, x_0, atol=1e-3)

        # At t=1, x_t should be almost purely y
        t_1 = torch.tensor([1.0, 1.0])
        x_t1, _ = sde.q_sample(x_0, y, t_1, noise=noise_zero)
        assert torch.allclose(x_t1, y, atol=1e-3)

    def test_fast_ode_sampling(self):
        sde = IRSDE(num_timesteps=20)
        unet = SARConditionalUNet(sar_channels=2, liss4_channels=3, out_channels=3, features=[16, 32, 64, 64], enable_time=True)
        unet.eval()

        cloudy = torch.rand(1, 3, 32, 32)
        sar = torch.rand(1, 2, 32, 32)

        out_ode = sde.sample_ode(unet, y=cloudy, sar=sar, steps=10)
        assert out_ode.shape == (1, 3, 32, 32)
        assert torch.isfinite(out_ode).all()
        assert (out_ode >= 0.0).all() and (out_ode <= 1.0).all()

    def test_sde_sampling(self):
        sde = IRSDE(num_timesteps=20)
        unet = SARConditionalUNet(sar_channels=2, liss4_channels=3, out_channels=3, features=[16, 32, 64, 64], enable_time=True)
        unet.eval()

        cloudy = torch.rand(1, 3, 32, 32)
        sar = torch.rand(1, 2, 32, 32)

        out_sde = sde.sample_sde(unet, y=cloudy, sar=sar, steps=10, eta=0.2)
        assert out_sde.shape == (1, 3, 32, 32)
        assert torch.isfinite(out_sde).all()


class TestSARDiffusionWrapperIRSDE:
    def test_forward_fast_ode(self):
        model = SARDiffusionWrapper(sar_channels=2, liss4_channels=3, out_channels=3, use_ir_sde=True)
        liss4 = torch.rand(1, 3, 32, 32)
        sar = torch.rand(1, 2, 32, 32)
        out = model(liss4, sar, fast=True, steps=10)
        assert out.shape == (1, 3, 32, 32)

    def test_forward_with_dem(self):
        model = SARDiffusionWrapper(sar_channels=2, liss4_channels=3, out_channels=3, dem_channels=4, use_ir_sde=True)
        liss4 = torch.rand(1, 3, 32, 32)
        sar = torch.rand(1, 2, 32, 32)
        dem = torch.rand(1, 4, 32, 32)
        out = model(liss4, sar, dem=dem, fast=True, steps=5)
        assert out.shape == (1, 3, 32, 32)

    def test_compute_loss_gradient_flow(self):
        model = SARDiffusionWrapper(sar_channels=2, liss4_channels=3, out_channels=3, use_ir_sde=True)
        clear = torch.rand(2, 3, 32, 32, requires_grad=False)
        cloudy = torch.rand(2, 3, 32, 32, requires_grad=False)
        sar = torch.rand(2, 2, 32, 32, requires_grad=False)
        mask = torch.ones(2, 1, 32, 32)

        loss = model.compute_loss(clear, cloudy, sar, cloud_mask=mask)
        assert torch.isfinite(loss)
        loss.backward()

        grads = [p.grad for p in model.parameters() if p.requires_grad]
        assert any(g is not None and g.abs().sum() > 0 for g in grads)
