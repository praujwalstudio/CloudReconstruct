import torch
import numpy as np
from pathlib import Path


class TestFullPipeline:
    def test_pipeline_with_synthetic_data(self, tmp_path):
        from src.training.synthetic_data import generate_synthetic_patches
        from src.evaluation.inference import CloudFreeInference

        patch_dir = generate_synthetic_patches(tmp_path, n_scenes=2, patches_per_scene=3,
                                                with_sar=False, with_temporal_ref=False)
        assert patch_dir.exists()

        fake_image = np.random.randint(0, 1023, (256, 256, 3), dtype=np.uint16)
        model = CloudFreeInference(device="cpu")
        result = model.correct(fake_image, data_max=1023.0)

        assert "corrected" in result
        assert "density" in result
        assert "confidence" in result
        assert "ars" in result
        assert result["corrected"].shape == fake_image.shape
        assert result["density"].ndim == 2
        assert result["confidence"].ndim == 2
        assert 0 <= result["density"].min() <= result["density"].max() <= 1
        assert 0 <= result["confidence"].min() <= result["confidence"].max() <= 1

    def test_pipeline_with_sar(self, tmp_path):
        from src.training.synthetic_data import generate_synthetic_patches
        from src.evaluation.inference import CloudFreeInference

        generate_synthetic_patches(tmp_path, n_scenes=1, patches_per_scene=2,
                                    with_sar=False, with_temporal_ref=False)

        fake_liss4 = np.random.randint(0, 1023, (64, 64, 3), dtype=np.uint16)
        fake_sar = np.random.randint(0, 500, (64, 64, 2), dtype=np.uint16)
        model = CloudFreeInference(device="cpu")
        result = model.correct(fake_liss4, sar=fake_sar, data_max=1023.0)

        assert result["corrected"].shape == fake_liss4.shape

    def test_pipeline_with_temporal_refs(self, tmp_path):
        from src.training.synthetic_data import generate_synthetic_patches
        from src.evaluation.inference import CloudFreeInference

        generate_synthetic_patches(tmp_path, n_scenes=1, patches_per_scene=2,
                                    with_sar=False, with_temporal_ref=False)

        fake_liss4 = np.random.randint(0, 1023, (64, 64, 3), dtype=np.uint16)
        fake_ref = np.random.randint(0, 1023, (64, 64, 3), dtype=np.uint16)
        model = CloudFreeInference(device="cpu")
        result = model.correct(fake_liss4, temporal_refs=[fake_ref], data_max=1023.0)

        assert result["corrected"].shape == fake_liss4.shape

    def test_pipeline_with_metrics(self, tmp_path):
        from src.evaluation.inference import CloudFreeInference
        from src.evaluation.metrics import compute_all_metrics

        fake_pred = np.random.rand(64, 64, 3).astype(np.float32)
        fake_target = np.random.rand(64, 64, 3).astype(np.float32)

        metrics = compute_all_metrics(fake_pred, fake_target)
        for key in ["psnr", "ssim", "sam", "ndvi_correlation"]:
            assert key in metrics, f"Missing metric: {key}"
        assert metrics["ssim"] is not None, "SSIM should not be None"

    def test_train_all_scripts_importable(self):
        from src.training.train_density import DensityTrainer, PatchDataset
        from src.training.train_correction import CorrectionTrainer
        from src.training.train_temporal import TemporalTrainer, TemporalPairDataset, create_temporal_dataloaders
        from src.training.train_diffusion import DiffusionTrainer, DiffusionSchedule
        from src.training.train_all import train_density, train_correction, train_temporal, train_diffusion
        assert True

    def test_density_trainer_creates_checkpoint(self, tmp_path):
        from src.models.cloud_density import CloudDensityNet
        from src.training.train_density import DensityTrainer
        from torch.utils.data import TensorDataset, DataLoader

        model = CloudDensityNet(in_channels=3, out_channels=1, dropout_p=0.0)
        trainer = DensityTrainer(model)

        images = torch.randn(4, 3, 64, 64)
        targets = torch.rand(4, 1, 64, 64)
        ds = TensorDataset(images, targets)
        loader = DataLoader(ds, batch_size=2)

        trainer.fit(loader, loader, epochs=2, checkpoint_dir=tmp_path)
        assert (tmp_path / "best_model.pth").exists()

    def test_mc_dropout_uncertainty(self):
        from src.models.cloud_density import CloudDensityNet

        model = CloudDensityNet(in_channels=3, out_channels=1, dropout_p=0.1)
        x = torch.randn(1, 3, 64, 64)
        mean, std = model.predict_with_uncertainty(x, n_samples=5)
        assert mean.shape == (1, 1, 64, 64)
        assert std.shape == (1, 1, 64, 64)
        assert (std >= 0).all()

    def test_confidence_with_uncertainty(self):
        from src.evaluation.confidence import compute_confidence
        import numpy as np

        density = np.random.rand(64, 64).astype(np.float32)
        unc = np.random.rand(64, 64).astype(np.float32) * 0.5

        conf = compute_confidence(density, uncertainty=unc)
        assert conf.shape == density.shape
        assert 0 <= conf.min() <= conf.max() <= 1

    def test_five_class_cloud_detection(self):
        from src.preprocessing.cloud_mask import classify_cloud_density, compute_cloud_shadow_mask

        density = np.random.rand(64, 64).astype(np.float32)
        fake_image = np.random.rand(64, 64, 3).astype(np.float32)
        fake_image[..., 2] = 0.05

        shadow = compute_cloud_shadow_mask(fake_image)
        classes = classify_cloud_density(density, shadow)
        assert classes.dtype == np.uint8
        assert classes.shape == density.shape
        assert set(np.unique(classes)).issubset({0, 1, 2, 3, 4})

    def test_cross_attention_forward(self):
        from src.models.sar_fusion import SARConditionalUNet

        model = SARConditionalUNet(sar_channels=2, liss4_channels=3, out_channels=3,
                                    enable_attention=True)
        liss4 = torch.randn(1, 3, 64, 64)
        sar = torch.randn(1, 2, 64, 64)
        out = model(liss4, sar)
        assert out.shape == (1, 3, 64, 64)

    def test_self_attention_temporal_forward(self):
        from src.models.temporal_fusion import TemporalFusion

        model = TemporalFusion(in_channels=3)
        cloudy = torch.randn(1, 3, 32, 32)
        ref = torch.randn(1, 3, 32, 32)
        density = torch.rand(1, 1, 32, 32)
        out = model(cloudy, ref, density)
        assert out.shape == (1, 3, 32, 32)
