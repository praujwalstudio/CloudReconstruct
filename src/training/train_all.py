import argparse
import yaml
import torch
from pathlib import Path

from src.config import PATCHES, CHECKPOINTS, SEN12MS_RAW, SEN12MS_COMPACT
from src.training.train_density import DensityTrainer, create_dataloaders
from src.training.train_correction import CorrectionTrainer
from src.training.train_temporal import TemporalTrainer, create_temporal_dataloaders
from src.training.train_diffusion import DiffusionTrainer, DiffusionSchedule


def get_patch_dir() -> Path:
    if SEN12MS_COMPACT.exists() and any(SEN12MS_COMPACT.rglob("*.npz")):
        return SEN12MS_COMPACT
    if SEN12MS_RAW.exists() and any(SEN12MS_RAW.rglob("*.tif*")):
        return SEN12MS_RAW
    return SEN12MS_COMPACT


def train_density(args):
    from src.models.cloud_density import CloudDensityNet
    print("\n" + "=" * 60)
    print("Training CloudDensityNet (with DEM Early Fusion & AMP support)")
    print("=" * 60)
    model = CloudDensityNet(in_channels=3, out_channels=1, dem_channels=args.dem_channels)
    trainer = DensityTrainer(model, device=args.device, use_amp=args.use_amp, use_fjl=args.use_fjl)
    loader = create_dataloaders(args.patch_dir, batch_size=args.batch_size, dataset_type=args.dataset_type)
    trainer.fit(
        loader[0], loader[1], epochs=args.epochs,
        checkpoint_dir=CHECKPOINTS / "density_model",
        resume_from=args.resume_from,
    )


def train_correction(args):
    from src.models.thin_cloud_correction import ThinCloudCorrection
    from src.models.cloud_density import CloudDensityNet
    from src.models.discriminator import PatchGANDiscriminator
    print("\n" + "=" * 60)
    print("Training ThinCloudCorrection (with Spatial Attention & GAN options)")
    print("=" * 60)
    model = ThinCloudCorrection(in_channels=3)
    discriminator = None
    if getattr(args, "use_adv", False):
        discriminator = PatchGANDiscriminator(in_channels=3)
        print("  Enabled PatchGAN Discriminator for adversarial loss.")

    density_model = None
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.with_density:
        density_ckpt = CHECKPOINTS / "density_model" / "best_model.pth"
        if density_ckpt.exists():
            density_model = CloudDensityNet(in_channels=3, out_channels=1, dem_channels=args.dem_channels)
            density_model.load_state_dict(
                torch.load(density_ckpt, map_location=device, weights_only=False)
            )
            density_model.to(device)
            density_model.eval()
            print(f"  Loaded density model from {density_ckpt}")
    trainer = CorrectionTrainer(model, discriminator=discriminator, device=device, use_amp=args.use_amp)
    loader = create_dataloaders(args.patch_dir, batch_size=args.batch_size, dataset_type=args.dataset_type)
    trainer.fit(
        loader[0], loader[1], epochs=args.epochs,
        checkpoint_dir=CHECKPOINTS / "correction_model",
        density_model=density_model,
        resume_from=args.resume_from,
    )


def train_temporal(args):
    from src.models.temporal_fusion import TemporalFusion
    print("\n" + "=" * 60)
    print("Training TemporalFusion (with MultiScaleConvergenceLoss & Uncertainty support)")
    print("=" * 60)
    model = TemporalFusion(in_channels=3)
    trainer = TemporalTrainer(
        model,
        device=args.device,
        use_amp=args.use_amp,
        use_multiscale_loss=args.use_multiscale_loss,
        use_uncertainty=getattr(args, "use_uncertainty", True),
    )
    loader = create_temporal_dataloaders(args.patch_dir, batch_size=args.batch_size)
    trainer.fit(
        loader[0], loader[1], epochs=args.epochs,
        checkpoint_dir=CHECKPOINTS / "temporal_model",
        resume_from=args.resume_from,
    )


def train_diffusion(args):
    from src.models.sar_fusion import SARDiffusionWrapper
    print("\n" + "=" * 60)
    print("Training SARDiffusionWrapper (with Mean-Reverting IR-SDE & DEM support)")
    print("=" * 60)
    model = SARDiffusionWrapper(
        sar_channels=2, liss4_channels=3, out_channels=3,
        noise_steps=args.noise_steps, dem_channels=args.dem_channels
    )
    trainer = DiffusionTrainer(model, device=args.device, use_amp=args.use_amp)
    loader = create_dataloaders(args.patch_dir, batch_size=args.batch_size, dataset_type=args.dataset_type)
    trainer.fit(
        loader[0], loader[1], epochs=args.epochs,
        checkpoint_dir=CHECKPOINTS / "diffusion_model",
        resume_from=args.resume_from,
    )


def main():
    parser = argparse.ArgumentParser(description="Train all CloudReconstruct models with Multi-Modal & AMP support")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML configuration file (e.g. configs/colab.yaml)")
    parser.add_argument("--model", type=str, default="all",
                        choices=["all", "density", "correction", "temporal", "diffusion"],
                        help="Which model to train")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--noise-steps", type=int, default=100, help="Diffusion noise steps")
    parser.add_argument("--dem-channels", type=int, default=4, help="Topographic DEM channels (elevation, slope, aspect, hillshade)")
    parser.add_argument("--use-amp", action="store_true", default=None, help="Enable Automatic Mixed Precision (torch.amp)")
    parser.add_argument("--use-fjl", action="store_true", help="Enable Filtered Jaccard Loss for density model")
    parser.add_argument("--use-adv", action="store_true", help="Enable PatchGAN adversarial loss for correction model")
    parser.add_argument("--use-uncertainty", action="store_true", default=True, help="Enable heteroscedastic uncertainty head in temporal model")
    parser.add_argument("--use-multiscale-loss", action="store_true", default=True,
                        help="Enable MultiScaleConvergenceLoss (L1 + SSIM + SAM) for temporal & diffusion models")
    parser.add_argument("--resume-from", type=str, default=None, help="Path to checkpoint for auto-resuming training")
    parser.add_argument("--dataset-type", type=str, default="auto", choices=["auto", "sen12ms_cr", "patches"],
                        help="Dataset format to load")
    parser.add_argument("--with-density", action="store_true",
                        help="Use pretrained density model for correction training")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    args = parser.parse_args()

    if args.config and Path(args.config).exists():
        print(f"[CONFIG] Loading configuration from {args.config}")
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
            if "training" in cfg:
                args.batch_size = cfg["training"].get("batch_size", args.batch_size)
                args.epochs = cfg["training"].get("epochs", args.epochs)
                if args.use_amp is None:
                    args.use_amp = cfg["training"].get("use_amp", None)
            if "data" in cfg:
                args.dataset_type = cfg["data"].get("dataset_type", args.dataset_type)
                args.dem_channels = cfg["data"].get("dem_channels", args.dem_channels)
            if "checkpoints" in cfg:
                resume_path = cfg["checkpoints"].get("resume_checkpoint")
                if resume_path and Path(resume_path).exists():
                    args.resume_from = resume_path

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.use_amp is None:
        args.use_amp = True if "cuda" in args.device else False

    if args.device.startswith("cuda"):
        if ":" in args.device:
            torch.cuda.set_device(int(args.device.split(":")[1]))
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Unknown GPU"
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3) if torch.cuda.is_available() else 0.0
        print(f"[COMPUTE] Device: {args.device} ({gpu_name}, {vram_gb:.1f} GB VRAM) | AMP Enabled: {args.use_amp}")
    else:
        print(f"[COMPUTE] Device: CPU | AMP Enabled: {args.use_amp}")

    args.patch_dir = get_patch_dir()

    model_map = {
        "density": train_density,
        "correction": train_correction,
        "temporal": train_temporal,
        "diffusion": train_diffusion,
    }

    if args.model == "all":
        for name, fn in model_map.items():
            fn(args)
    else:
        model_map[args.model](args)

    print("\n" + "=" * 60)
    print("Training complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
