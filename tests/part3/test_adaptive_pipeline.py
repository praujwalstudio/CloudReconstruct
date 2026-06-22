import torch


class TestAdaptivePipeline:
    def test_import(self):
        from src.models.adaptive_pipeline import AdaptiveCloudRemoval
        assert AdaptiveCloudRemoval is not None

    def test_forward_with_all_modules(self):
        from src.models.cloud_density import CloudDensityNet
        from src.models.thin_cloud_correction import ThinCloudCorrection
        from src.models.temporal_fusion import TemporalFusion
        from src.models.sar_fusion import SARConditionalUNet
        from src.models.adaptive_pipeline import AdaptiveCloudRemoval

        density_net = CloudDensityNet(in_channels=3, out_channels=1)
        correction_net = ThinCloudCorrection(in_channels=3)
        temporal_fusion = TemporalFusion(in_channels=3)
        sar_fusion = SARConditionalUNet(sar_channels=2, liss4_channels=3, out_channels=3)

        model = AdaptiveCloudRemoval(
            density_net=density_net,
            correction_net=correction_net,
            temporal_fusion=temporal_fusion,
            sar_fusion=sar_fusion,
        )

        liss4 = torch.randn(2, 3, 64, 64)
        sar = torch.randn(2, 2, 64, 64)
        refs = [torch.randn(2, 3, 64, 64)]
        output, density, weights = model(liss4, sar=sar, temporal_refs=refs)

        assert output.shape == (2, 3, 64, 64)
        assert density.shape == (2, 1, 64, 64)
        assert weights.shape == (2, 3, 64, 64)

    def test_forward_without_sar(self):
        from src.models.cloud_density import CloudDensityNet
        from src.models.thin_cloud_correction import ThinCloudCorrection
        from src.models.temporal_fusion import TemporalFusion
        from src.models.sar_fusion import SARConditionalUNet
        from src.models.adaptive_pipeline import AdaptiveCloudRemoval

        density_net = CloudDensityNet(in_channels=3, out_channels=1)
        correction_net = ThinCloudCorrection(in_channels=3)
        temporal_fusion = TemporalFusion(in_channels=3)
        sar_fusion = SARConditionalUNet(sar_channels=2, liss4_channels=3, out_channels=3)

        model = AdaptiveCloudRemoval(density_net, correction_net, temporal_fusion, sar_fusion)
        liss4 = torch.randn(1, 3, 32, 32)
        output, density, weights = model(liss4)
        assert output.shape == (1, 3, 32, 32)
        assert density.shape == (1, 1, 32, 32)
        assert weights.shape == (1, 3, 32, 32)

    def test_forward_without_temporal(self):
        from src.models.cloud_density import CloudDensityNet
        from src.models.thin_cloud_correction import ThinCloudCorrection
        from src.models.temporal_fusion import TemporalFusion
        from src.models.sar_fusion import SARConditionalUNet
        from src.models.adaptive_pipeline import AdaptiveCloudRemoval

        density_net = CloudDensityNet(in_channels=3, out_channels=1)
        correction_net = ThinCloudCorrection(in_channels=3)
        temporal_fusion = TemporalFusion(in_channels=3)
        sar_fusion = SARConditionalUNet(sar_channels=2, liss4_channels=3, out_channels=3)

        model = AdaptiveCloudRemoval(density_net, correction_net, temporal_fusion, sar_fusion)
        liss4 = torch.randn(1, 3, 32, 32)
        output, density, weights = model(liss4, sar=torch.randn(1, 2, 32, 32))
        assert output.shape == (1, 3, 32, 32)

    def test_gradient_flow(self):
        from src.models.cloud_density import CloudDensityNet
        from src.models.thin_cloud_correction import ThinCloudCorrection
        from src.models.temporal_fusion import TemporalFusion
        from src.models.sar_fusion import SARConditionalUNet
        from src.models.adaptive_pipeline import AdaptiveCloudRemoval

        density_net = CloudDensityNet(in_channels=3, out_channels=1)
        correction_net = ThinCloudCorrection(in_channels=3)
        temporal_fusion = TemporalFusion(in_channels=3)
        sar_fusion = SARConditionalUNet(sar_channels=2, liss4_channels=3, out_channels=3)

        model = AdaptiveCloudRemoval(density_net, correction_net, temporal_fusion, sar_fusion)
        liss4 = torch.randn(2, 3, 32, 32)
        target = torch.randn(2, 3, 32, 32)
        output, _, _ = model(liss4, sar=torch.randn(2, 2, 32, 32),
                             temporal_refs=[torch.randn(2, 3, 32, 32)])
        loss = torch.nn.functional.mse_loss(output, target)
        loss.backward()
        assert all(p.grad is not None for p in density_net.parameters() if p.requires_grad)
        assert all(p.grad is not None for p in correction_net.parameters() if p.requires_grad)


class TestBuildAdaptivePipeline:
    def test_build_returns_model(self):
        from src.models.adaptive_pipeline import build_adaptive_pipeline
        model = build_adaptive_pipeline()
        from src.models.adaptive_pipeline import AdaptiveCloudRemoval
        assert isinstance(model, AdaptiveCloudRemoval)

    def test_build_with_custom_thresholds(self):
        from src.models.adaptive_pipeline import build_adaptive_pipeline
        model = build_adaptive_pipeline(thin_threshold=0.2, medium_threshold=0.4, dense_threshold=0.7)
        assert model.thin_threshold == 0.2
        assert model.medium_threshold == 0.4
        assert model.dense_threshold == 0.7


class TestBlendWeights:
    def test_weight_sum_to_one(self):
        from src.models.adaptive_pipeline import AdaptiveCloudRemoval
        from src.models.cloud_density import CloudDensityNet
        from src.models.thin_cloud_correction import ThinCloudCorrection
        from src.models.temporal_fusion import TemporalFusion
        from src.models.sar_fusion import SARConditionalUNet

        density_net = CloudDensityNet(in_channels=3, out_channels=1)
        model = AdaptiveCloudRemoval(
            density_net, ThinCloudCorrection(3), TemporalFusion(3), SARConditionalUNet(2, 3, 3)
        )
        density = torch.tensor([[[[0.1, 0.4, 0.6, 0.9]]]], dtype=torch.float32)
        weights = model._compute_blend_weights(density)
        assert weights.shape == (1, 3, 1, 4)
        assert torch.allclose(weights.sum(dim=1), torch.ones(1, 1, 4), atol=1e-5)

    def test_clear_pixel_uses_thin_path(self):
        from src.models.adaptive_pipeline import AdaptiveCloudRemoval
        from src.models.cloud_density import CloudDensityNet
        from src.models.thin_cloud_correction import ThinCloudCorrection
        from src.models.temporal_fusion import TemporalFusion
        from src.models.sar_fusion import SARConditionalUNet

        density_net = CloudDensityNet(in_channels=3, out_channels=1)
        model = AdaptiveCloudRemoval(
            density_net, ThinCloudCorrection(3), TemporalFusion(3), SARConditionalUNet(2, 3, 3)
        )

        density = torch.zeros(1, 1, 4, 4)
        weights = model._compute_blend_weights(density)
        assert weights[0, 0].mean() > 0.5, "Clear pixels should route to thin path"
