import argparse
import torch
from pathlib import Path

from src.config import PATCHES, CHECKPOINTS
from src.training.train_density import DensityTrainer, create_dataloaders
from src.training.train_correction import CorrectionTrainer
from src.training.train_temporal import TemporalTrainer, create_temporal_dataloaders
from src.training.train_diffusion import DiffusionTrainer, DiffusionSchedule
from src.training.synthetic_data import generate_synthetic_patches


def get_patch_dir() -> Path:
    patch_dir = PATCHES
    if not patch_dir.exists() or not any(patch_dir.rglob("*.npy")):
        print("[DATA] No patches found. Generating synthetic data...")
        patch_dir = generate_synthetic_patches()
    return patch_dir


def train_density(args):
    from src.models.cloud_density import CloudDensityNet
    print("\n" + "=" * 60)
    print("Training CloudDensityNet")
    print("=" * 60)
    model = CloudDensityNet(in_channels=3, out_channels=1)
    trainer = DensityTrainer(model)
    loader = create_dataloaders(args.patch_dir, batch_size=args.batch_size)
    trainer.fit(loader[0], loader[1], epochs=args.epochs,
                checkpoint_dir=CHECKPOINTS / "density_model")


def train_correction(args):
    from src.models.thin_cloud_correction import ThinCloudCorrection
    from src.models.cloud_density import CloudDensityNet
    print("\n" + "=" * 60)
    print("Training ThinCloudCorrection")
    print("=" * 60)
    model = ThinCloudCorrection(in_channels=3)
    density_model = None
    if args.with_density:
        density_ckpt = CHECKPOINTS / "density_model" / "best_model.pth"
        if density_ckpt.exists():
            density_model = CloudDensityNet(in_channels=3, out_channels=1)
            density_model.load_state_dict(
                torch.load(density_ckpt, map_location="cpu", weights_only=False)
            )
            print(f"  Loaded density model from {density_ckpt}")
    trainer = CorrectionTrainer(model)
    loader = create_dataloaders(args.patch_dir, batch_size=args.batch_size)
    trainer.fit(loader[0], loader[1], epochs=args.epochs,
                checkpoint_dir=CHECKPOINTS / "correction_model",
                density_model=density_model)


def train_temporal(args):
    from src.models.temporal_fusion import TemporalFusion
    print("\n" + "=" * 60)
    print("Training TemporalFusion")
    print("=" * 60)
    model = TemporalFusion(in_channels=3)
    trainer = TemporalTrainer(model)
    loader = create_temporal_dataloaders(args.patch_dir, batch_size=args.batch_size)
    trainer.fit(loader[0], loader[1], epochs=args.epochs,
                checkpoint_dir=CHECKPOINTS / "temporal_model")


def train_diffusion(args):
    from src.models.sar_fusion import SARDiffusionWrapper
    print("\n" + "=" * 60)
    print("Training SARDiffusionWrapper")
    print("=" * 60)
    model = SARDiffusionWrapper(sar_channels=2, liss4_channels=3, out_channels=3,
                                noise_steps=args.noise_steps)
    trainer = DiffusionTrainer(model)
    loader = create_dataloaders(args.patch_dir, batch_size=args.batch_size)
    trainer.fit(loader[0], loader[1], epochs=args.epochs,
                checkpoint_dir=CHECKPOINTS / "diffusion_model")


def main():
    parser = argparse.ArgumentParser(description="Train all CloudReconstruct models")
    parser.add_argument("--model", type=str, default="all",
                        choices=["all", "density", "correction", "temporal", "diffusion"],
                        help="Which model to train")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--noise-steps", type=int, default=100, help="Diffusion noise steps")
    parser.add_argument("--with-density", action="store_true",
                        help="Use pretrained density model for correction training")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    args = parser.parse_args()

    if args.device:
        torch.cuda.set_device(args.device) if args.device.startswith("cuda") else None

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
