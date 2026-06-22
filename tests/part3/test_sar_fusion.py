import torch


class TestSARConditionalUNet:
    def test_forward_shape(self):
        from src.models.sar_fusion import SARConditionalUNet
        model = SARConditionalUNet(sar_channels=2, liss4_channels=3, out_channels=3)
        liss4 = torch.randn(2, 3, 256, 256)
        sar = torch.randn(2, 2, 256, 256)
        out = model(liss4, sar)
        assert out.shape == (2, 3, 256, 256)

    def test_output_range(self):
        from src.models.sar_fusion import SARConditionalUNet
        model = SARConditionalUNet(sar_channels=2, liss4_channels=3, out_channels=3)
        model.eval()
        with torch.no_grad():
            liss4 = torch.randn(1, 3, 64, 64)
            sar = torch.randn(1, 2, 64, 64)
            out = model(liss4, sar)
            assert out.shape == (1, 3, 64, 64)

    def test_gradient_flow(self):
        from src.models.sar_fusion import SARConditionalUNet
        model = SARConditionalUNet(sar_channels=2, liss4_channels=3, out_channels=3)
        liss4 = torch.randn(2, 3, 64, 64)
        sar = torch.randn(2, 2, 64, 64)
        target = torch.randn(2, 3, 64, 64)
        out = model(liss4, sar)
        loss = torch.nn.functional.mse_loss(out, target)
        loss.backward()
        assert all(p.grad is not None for p in model.parameters() if p.requires_grad)

    def test_different_sar_channels(self):
        from src.models.sar_fusion import SARConditionalUNet
        for sar_ch in [1, 2, 4]:
            model = SARConditionalUNet(sar_channels=sar_ch, liss4_channels=3, out_channels=3)
            liss4 = torch.randn(1, 3, 32, 32)
            sar = torch.randn(1, sar_ch, 32, 32)
            out = model(liss4, sar)
            assert out.shape == (1, 3, 32, 32), f"Failed at sar_channels={sar_ch}"

    def test_overfit_single_batch(self):
        from src.models.sar_fusion import SARConditionalUNet
        model = SARConditionalUNet(sar_channels=2, liss4_channels=3, out_channels=3)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        liss4 = torch.randn(4, 3, 32, 32)
        sar = torch.randn(4, 2, 32, 32)
        target = torch.randn(4, 3, 32, 32)

        initial_loss = None
        for _ in range(100):
            optimizer.zero_grad()
            out = model(liss4, sar)
            loss = torch.nn.functional.mse_loss(out, target)
            if initial_loss is None:
                initial_loss = loss.item()
            loss.backward()
            optimizer.step()

        final_loss = loss.item()
        assert final_loss < initial_loss, (
            f"Loss did not decrease: {initial_loss:.6f} -> {final_loss:.6f}"
        )

    def test_state_dict_save_load(self, tmp_path):
        from src.models.sar_fusion import SARConditionalUNet
        model = SARConditionalUNet(sar_channels=2, liss4_channels=3, out_channels=3)
        model.eval()
        liss4 = torch.randn(1, 3, 32, 32)
        sar = torch.randn(1, 2, 32, 32)
        with torch.no_grad():
            out_before = model(liss4, sar)

        ckpt = tmp_path / "sar_model.pth"
        torch.save(model.state_dict(), ckpt)

        model2 = SARConditionalUNet(sar_channels=2, liss4_channels=3, out_channels=3)
        model2.load_state_dict(torch.load(ckpt))
        model2.eval()
        with torch.no_grad():
            out_after = model2(liss4, sar)

        assert torch.allclose(out_before, out_after, atol=1e-6)


class TestSARDiffusionWrapper:
    def test_forward_shape(self):
        from src.models.sar_fusion import SARDiffusionWrapper
        model = SARDiffusionWrapper(sar_channels=2, liss4_channels=3, out_channels=3)
        liss4 = torch.randn(1, 3, 64, 64)
        sar = torch.randn(1, 2, 64, 64)
        out = model(liss4, sar)
        assert out.shape == (1, 3, 64, 64)

    def test_noise_steps_attribute(self):
        from src.models.sar_fusion import SARDiffusionWrapper
        model = SARDiffusionWrapper(sar_channels=2, liss4_channels=3, out_channels=3,
                                    noise_steps=200)
        assert model.noise_steps == 200
