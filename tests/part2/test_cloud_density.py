import torch
import numpy as np


class TestCloudDensityNet:
    def test_import(self):
        from src.models.cloud_density import CloudDensityNet
        assert CloudDensityNet is not None

    def test_forward_shape(self):
        from src.models.cloud_density import CloudDensityNet
        model = CloudDensityNet(in_channels=3, out_channels=1)
        x = torch.randn(4, 3, 256, 256)
        out = model(x)
        assert out.shape == (4, 1, 256, 256)

    def test_output_range(self):
        from src.models.cloud_density import CloudDensityNet
        model = CloudDensityNet(in_channels=3, out_channels=1)
        model.eval()
        with torch.no_grad():
            x = torch.randn(2, 3, 64, 64)
            out = model(x)
            assert out.min() >= 0.0
            assert out.max() <= 1.0

    def test_gradient_flow(self):
        from src.models.cloud_density import CloudDensityNet
        model = CloudDensityNet(in_channels=3, out_channels=1)
        x = torch.randn(2, 3, 64, 64)
        target = torch.rand(2, 1, 64, 64)
        out = model(x)
        loss = torch.nn.functional.mse_loss(out, target)
        loss.backward()
        assert all(p.grad is not None for p in model.parameters() if p.requires_grad)

    def test_overfit_single_batch(self):
        from src.models.cloud_density import CloudDensityNet
        model = CloudDensityNet(in_channels=3, out_channels=1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        x = torch.randn(4, 3, 64, 64)
        target = torch.rand(4, 1, 64, 64)

        initial_loss = None
        for _ in range(100):
            optimizer.zero_grad()
            out = model(x)
            loss = torch.nn.functional.mse_loss(out, target)
            if initial_loss is None:
                initial_loss = loss.item()
            loss.backward()
            optimizer.step()

        final_loss = loss.item()
        assert final_loss < initial_loss, (
            f"Loss did not decrease: {initial_loss:.6f} -> {final_loss:.6f}"
        )

    def test_different_input_sizes(self):
        from src.models.cloud_density import CloudDensityNet
        model = CloudDensityNet(in_channels=3, out_channels=1)
        for size in [128, 256, 512]:
            x = torch.randn(1, 3, size, size)
            out = model(x)
            assert out.shape == (1, 1, size, size), f"Failed at size {size}"

    def test_state_dict_save_load(self, tmp_path):
        from src.models.cloud_density import CloudDensityNet
        model = CloudDensityNet(in_channels=3, out_channels=1)
        model.eval()
        x = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            out_before = model(x)

        ckpt = tmp_path / "model.pth"
        torch.save(model.state_dict(), ckpt)

        model2 = CloudDensityNet(in_channels=3, out_channels=1)
        model2.load_state_dict(torch.load(ckpt))
        model2.eval()
        with torch.no_grad():
            out_after = model2(x)

        assert torch.allclose(out_before, out_after, atol=1e-6)


class TestDoubleConv:
    def test_forward_shape(self):
        from src.models.cloud_density import DoubleConv
        block = DoubleConv(3, 32)
        x = torch.randn(2, 3, 64, 64)
        out = block(x)
        assert out.shape == (2, 32, 64, 64)


class TestDown:
    def test_downsample_shape(self):
        from src.models.cloud_density import Down
        block = Down(32, 64)
        x = torch.randn(2, 32, 64, 64)
        out = block(x)
        assert out.shape == (2, 64, 32, 32)


class TestUp:
    def test_upsample_shape(self):
        from src.models.cloud_density import Up
        block = Up(128, 64)
        x1 = torch.randn(2, 64, 16, 16)
        x2 = torch.randn(2, 64, 32, 32)
        out = block(x1, x2)
        assert out.shape == (2, 64, 32, 32)
