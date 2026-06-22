import torch


class TestThinCloudCorrection:
    def test_import(self):
        from src.models.thin_cloud_correction import ThinCloudCorrection
        assert ThinCloudCorrection is not None

    def test_forward_shape(self):
        from src.models.thin_cloud_correction import ThinCloudCorrection
        model = ThinCloudCorrection(in_channels=3)
        cloudy = torch.randn(4, 3, 256, 256)
        density = torch.rand(4, 1, 256, 256)
        out = model(cloudy, density)
        assert out.shape == (4, 3, 256, 256)

    def test_gradient_flow(self):
        from src.models.thin_cloud_correction import ThinCloudCorrection
        model = ThinCloudCorrection(in_channels=3)
        cloudy = torch.randn(2, 3, 64, 64)
        density = torch.rand(2, 1, 64, 64)
        target = torch.randn(2, 3, 64, 64)
        out = model(cloudy, density)
        loss = torch.nn.functional.mse_loss(out, target)
        loss.backward()
        assert all(p.grad is not None for p in model.parameters() if p.requires_grad)

    def test_identity_for_clear_pixels(self):
        from src.models.thin_cloud_correction import ThinCloudCorrection
        model = ThinCloudCorrection(in_channels=3)
        model.eval()
        with torch.no_grad():
            cloudy = torch.randn(1, 3, 64, 64)
            density = torch.zeros(1, 1, 64, 64)
            out = model(cloudy, density)
            assert out.shape == cloudy.shape

    def test_different_density_produces_different_output(self):
        from src.models.thin_cloud_correction import ThinCloudCorrection
        model = ThinCloudCorrection(in_channels=3)
        model.eval()
        with torch.no_grad():
            cloudy = torch.ones(1, 3, 32, 32)
            low_density = torch.zeros(1, 1, 32, 32)
            high_density = torch.ones(1, 1, 32, 32)

            out_low = model(cloudy, low_density)
            out_high = model(cloudy, high_density)

            assert not torch.allclose(out_low, out_high, atol=1e-4)


class TestCloudCorrectionPipeline:
    def test_pipeline_forward(self):
        from src.models.cloud_density import CloudDensityNet
        from src.models.thin_cloud_correction import ThinCloudCorrection, CloudCorrectionPipeline

        density_net = CloudDensityNet(in_channels=3, out_channels=1)
        correction = ThinCloudCorrection(in_channels=3)
        pipeline = CloudCorrectionPipeline(density_net, correction)

        x = torch.randn(2, 3, 128, 128)
        corrected, density = pipeline(x)

        assert corrected.shape == (2, 3, 128, 128)
        assert density.shape == (2, 1, 128, 128)
        assert density.min() >= 0.0
        assert density.max() <= 1.0
