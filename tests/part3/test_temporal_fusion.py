import torch


class TestAlignmentNet:
    def test_forward_shape(self):
        from src.models.temporal_fusion import AlignmentNet
        net = AlignmentNet(in_channels=6)
        cloudy = torch.randn(2, 3, 64, 64)
        ref = torch.randn(2, 3, 64, 64)
        flow = net(cloudy, ref)
        assert flow.shape == (2, 2, 64, 64)

    def test_flow_range(self):
        from src.models.temporal_fusion import AlignmentNet
        net = AlignmentNet(in_channels=6)
        net.eval()
        with torch.no_grad():
            cloudy = torch.randn(1, 3, 32, 32)
            ref = torch.randn(1, 3, 32, 32)
            flow = net(cloudy, ref)
            assert flow.min() >= -10.0
            assert flow.max() <= 10.0

    def test_gradient_flow(self):
        from src.models.temporal_fusion import AlignmentNet
        net = AlignmentNet(in_channels=6)
        cloudy = torch.randn(2, 3, 32, 32)
        ref = torch.randn(2, 3, 32, 32)
        flow = net(cloudy, ref)
        loss = flow.mean()
        loss.backward()
        assert all(p.grad is not None for p in net.parameters() if p.requires_grad)


class TestApplyFlow:
    def test_output_shape(self):
        from src.models.temporal_fusion import apply_flow
        image = torch.randn(2, 3, 64, 64)
        flow = torch.randn(2, 2, 64, 64) * 0.5
        warped = apply_flow(image, flow)
        assert warped.shape == (2, 3, 64, 64)

    def test_zero_flow_identity(self):
        from src.models.temporal_fusion import apply_flow
        image = torch.randn(1, 3, 32, 32)
        flow = torch.zeros(1, 2, 32, 32)
        warped = apply_flow(image, flow)
        assert torch.allclose(image, warped, atol=1e-5)


class TestTemporalFusion:
    def test_forward_shape(self):
        from src.models.temporal_fusion import TemporalFusion
        model = TemporalFusion(in_channels=3, hidden=64)
        cloudy = torch.randn(2, 3, 64, 64)
        ref = torch.randn(2, 3, 64, 64)
        density = torch.rand(2, 1, 64, 64)
        out = model(cloudy, ref, density)
        assert out.shape == (2, 3, 64, 64)

    def test_gradient_flow(self):
        from src.models.temporal_fusion import TemporalFusion
        model = TemporalFusion(in_channels=3)
        cloudy = torch.randn(2, 3, 32, 32)
        ref = torch.randn(2, 3, 32, 32)
        density = torch.rand(2, 1, 32, 32)
        target = torch.randn(2, 3, 32, 32)
        out = model(cloudy, ref, density)
        loss = torch.nn.functional.mse_loss(out, target)
        loss.backward()
        assert all(p.grad is not None for p in model.parameters() if p.requires_grad)

    def test_density_aware_blending(self):
        from src.models.temporal_fusion import TemporalFusion
        model = TemporalFusion(in_channels=3, hidden=64)
        model.eval()
        with torch.no_grad():
            cloudy = torch.randn(1, 3, 32, 32)
            ref = torch.randn(1, 3, 32, 32)

            zero_density = torch.zeros(1, 1, 32, 32)
            high_density = torch.ones(1, 1, 32, 32)

            out_zero = model(cloudy, ref, zero_density)
            out_high = model(cloudy, ref, high_density)

            diff_zero = torch.abs(out_zero - cloudy).mean().item()
            diff_high = torch.abs(out_high - cloudy).mean().item()
            assert diff_high != diff_zero


class TestMultiTemporalFusion:
    def test_single_reference(self):
        from src.models.temporal_fusion import MultiTemporalFusion
        model = MultiTemporalFusion(in_channels=3)
        cloudy = torch.randn(1, 3, 32, 32)
        refs = [torch.randn(1, 3, 32, 32)]
        density = torch.rand(1, 1, 32, 32)
        out, fused_list = model(cloudy, refs, density)
        assert out.shape == (1, 3, 32, 32)
        assert len(fused_list) == 1

    def test_multiple_references(self):
        from src.models.temporal_fusion import MultiTemporalFusion
        model = MultiTemporalFusion(in_channels=3)
        cloudy = torch.randn(1, 3, 32, 32)
        refs = [torch.randn(1, 3, 32, 32) for _ in range(3)]
        density = torch.rand(1, 1, 32, 32)
        out, fused_list = model(cloudy, refs, density)
        assert out.shape == (1, 3, 32, 32)
        assert len(fused_list) == 3

    def test_no_reference_fallback(self):
        from src.models.temporal_fusion import MultiTemporalFusion
        model = MultiTemporalFusion(in_channels=3)
        cloudy = torch.randn(1, 3, 32, 32)
        density = torch.rand(1, 1, 32, 32)
        out, fused_list = model(cloudy, [], density)
        assert out.shape == (1, 3, 32, 32)
        assert len(fused_list) == 0
