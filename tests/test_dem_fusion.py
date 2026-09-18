import pytest
import torch
import numpy as np

from src.preprocessing.dem_features import compute_dem_feature_stack
from src.models.cloud_density import CloudDensityNet
from src.models.sar_fusion import SARConditionalUNet, SARDiffusionWrapper


class TestDEMFusion:
    def test_compute_dem_feature_stack_shapes(self):
        # 1-channel DEM tensor
        dem_t = torch.rand(2, 1, 32, 32) * 500.0
        stack_t = compute_dem_feature_stack(dem_t, resolution=5.8)
        assert stack_t.shape == (2, 4, 32, 32)
        assert (stack_t >= 0.0).all() and (stack_t <= 1.0).all()

        # Numpy DEM array
        dem_np = np.random.rand(32, 32) * 500.0
        stack_np = compute_dem_feature_stack(dem_np, resolution=5.8)
        assert stack_np.shape == (4, 32, 32)

    def test_cloud_density_dem_early_fusion(self):
        model = CloudDensityNet(in_channels=3, out_channels=1, dem_channels=4)
        x = torch.rand(2, 3, 32, 32)
        dem = torch.rand(2, 4, 32, 32)

        # 1. Forward without DEM (backward compatibility)
        out_no_dem = model(x)
        assert out_no_dem.shape == (2, 1, 32, 32)

        # 2. Forward with 4-channel DEM early fusion
        out_dem = model(x, dem=dem)
        assert out_dem.shape == (2, 1, 32, 32)

        # 3. Forward with 1-channel raw DEM (auto-converted)
        dem_1ch = torch.rand(2, 1, 32, 32)
        out_1ch = model(x, dem=dem_1ch)
        assert out_1ch.shape == (2, 1, 32, 32)

        # 4. Gradients flow through both image and DEM
        dem_grad = torch.rand(2, 4, 32, 32, requires_grad=True)
        out = model(x, dem=dem_grad)
        loss = out.sum()
        loss.backward()
        assert dem_grad.grad is not None

    def test_sar_conditional_unet_dem_fusion(self):
        model = SARConditionalUNet(sar_channels=2, liss4_channels=3, out_channels=3, dem_channels=4)
        liss4 = torch.rand(2, 3, 32, 32)
        sar = torch.rand(2, 2, 32, 32)
        dem = torch.rand(2, 4, 32, 32)

        # Forward without DEM
        out_no_dem = model(liss4, sar)
        assert out_no_dem.shape == (2, 3, 32, 32)

        # Forward with DEM
        out_dem = model(liss4, sar, dem=dem)
        assert out_dem.shape == (2, 3, 32, 32)

    def test_sar_diffusion_wrapper_dem_fusion(self):
        model = SARDiffusionWrapper(sar_channels=2, liss4_channels=3, out_channels=3, noise_steps=10, dem_channels=4)
        clear = torch.rand(2, 3, 32, 32)
        liss4 = torch.rand(2, 3, 32, 32)
        sar = torch.rand(2, 2, 32, 32)
        dem = torch.rand(2, 4, 32, 32)

        loss = model.compute_loss(clear, liss4, sar, dem=dem)
        assert loss.item() >= 0.0
        assert not torch.isnan(loss)
