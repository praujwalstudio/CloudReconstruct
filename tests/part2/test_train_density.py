import torch
import numpy as np
from pathlib import Path


class TestPatchDataset:
    def test_load_from_synthetic(self, tmp_path):
        from src.training.synthetic_data import generate_synthetic_patches
        from src.training.train_density import PatchDataset

        patch_dir = generate_synthetic_patches(tmp_path, n_scenes=3, patches_per_scene=5)
        ds = PatchDataset("train", patch_dir)
        assert len(ds) > 0

        image, target = ds[0]
        assert isinstance(image, torch.Tensor)
        assert isinstance(target, torch.Tensor)
        assert image.shape[0] == 3
        assert target.shape[0] == 1

    def test_normalization_range(self, tmp_path):
        from src.training.synthetic_data import generate_synthetic_patches
        from src.training.train_density import PatchDataset

        patch_dir = generate_synthetic_patches(tmp_path, n_scenes=3, patches_per_scene=5)
        ds = PatchDataset("train", patch_dir)
        image, _ = ds[0]
        assert image.min() >= 0.0
        assert image.max() <= 1.0


class TestDensityTrainer:
    def test_train_epoch(self, tmp_path):
        from src.models.cloud_density import CloudDensityNet
        from src.training.train_density import DensityTrainer
        from torch.utils.data import TensorDataset, DataLoader

        model = CloudDensityNet(in_channels=3, out_channels=1)
        trainer = DensityTrainer(model)

        images = torch.randn(8, 3, 64, 64)
        targets = torch.rand(8, 1, 64, 64)
        ds = TensorDataset(images, targets)
        loader = DataLoader(ds, batch_size=4)

        loss = trainer.train_epoch(loader)
        assert isinstance(loss, float)
        assert loss >= 0.0

    def test_validation(self, tmp_path):
        from src.models.cloud_density import CloudDensityNet
        from src.training.train_density import DensityTrainer
        from torch.utils.data import TensorDataset, DataLoader

        model = CloudDensityNet(in_channels=3, out_channels=1)
        trainer = DensityTrainer(model)

        images = torch.randn(8, 3, 64, 64)
        targets = torch.rand(8, 1, 64, 64)
        ds = TensorDataset(images, targets)
        loader = DataLoader(ds, batch_size=4)

        metrics = trainer.validate(loader)
        for key in ["mse", "mae", "rmse"]:
            assert key in metrics
            assert metrics[key] >= 0.0

    def test_fit_saves_checkpoint(self, tmp_path):
        from src.models.cloud_density import CloudDensityNet
        from src.training.train_density import DensityTrainer
        from torch.utils.data import TensorDataset, DataLoader

        model = CloudDensityNet(in_channels=3, out_channels=1)
        trainer = DensityTrainer(model)

        images = torch.randn(8, 3, 64, 64)
        targets = torch.rand(8, 1, 64, 64)
        ds = TensorDataset(images, targets)
        loader = DataLoader(ds, batch_size=4)

        trainer.fit(loader, loader, epochs=2, checkpoint_dir=tmp_path)
        assert (tmp_path / "best_model.pth").exists()
        assert (tmp_path / "final_model.pth").exists()

    def test_load_checkpoint(self, tmp_path):
        from src.models.cloud_density import CloudDensityNet
        from src.training.train_density import DensityTrainer

        model = CloudDensityNet(in_channels=3, out_channels=1)
        trainer = DensityTrainer(model)

        torch.save(model.state_dict(), tmp_path / "test_model.pth")
        trainer.load_checkpoint(tmp_path / "test_model.pth")

        x = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            out = trainer.model(x)
        assert out.shape == (1, 1, 64, 64)


class TestCreateDataloaders:
    def test_dataloader_shapes(self, tmp_path):
        from src.training.synthetic_data import generate_synthetic_patches
        from src.training.train_density import create_dataloaders

        patch_dir = generate_synthetic_patches(tmp_path, n_scenes=10, patches_per_scene=5)
        train_loader, val_loader, test_loader = create_dataloaders(patch_dir, batch_size=4)
        assert len(train_loader) > 0
        assert len(val_loader) > 0
        assert len(test_loader) > 0

        for images, targets in train_loader:
            assert images.shape[0] <= 4
            assert images.shape[1] == 3
            assert targets.shape[1] == 1
            break


class TestSyntheticData:
    def test_generation_creates_files(self, tmp_path):
        from src.training.synthetic_data import generate_synthetic_patches
        patch_dir = generate_synthetic_patches(tmp_path, n_scenes=10, patches_per_scene=10)

        for split in ["train", "val", "test"]:
            split_dir = patch_dir / split
            assert split_dir.exists(), f"Missing split dir: {split}"
            npy_files = list(split_dir.glob("*.npy"))
            assert len(npy_files) > 0, f"No npy files in {split}"

    def test_split_ratios(self, tmp_path):
        from src.training.synthetic_data import generate_synthetic_patches
        patch_dir = generate_synthetic_patches(tmp_path, n_scenes=10, patches_per_scene=20)

        train_count = len(list((patch_dir / "train").glob("*.npy")))
        val_count = len(list((patch_dir / "val").glob("*.npy")))
        test_count = len(list((patch_dir / "test").glob("*.npy")))
        total = train_count + val_count + test_count

        assert total == 200
        assert 0.7 <= train_count / total <= 0.9
