import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import PATCHES, CHECKPOINTS
from src.training.train_density import PatchDataset, create_dataloaders
from src.training.losses import CombinedLoss


class CorrectionTrainer:
    def __init__(self, model: nn.Module, device: str = None,
                 loss_weights: dict = None):
        self.model = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.loss_fn = CombinedLoss(loss_weights or {}, device=self.device)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )

    def train_epoch(self, loader: DataLoader, density_model: nn.Module = None) -> dict:
        self.model.train()
        total_losses = {}

        for images, density_targets in tqdm(loader, desc="Training", leave=False):
            images = images.to(self.device)
            if images.shape[1] == 1:
                images = images.repeat(1, 3, 1, 1)

            cloudy = images
            density = density_targets.to(self.device) if density_targets is not None else None
            if density is None and density_model is not None:
                with torch.no_grad():
                    density = density_model(cloudy)

            correction = self.model(cloudy, density if density is not None else torch.zeros_like(cloudy[:, :1]))
            corrected = cloudy + torch.clamp(correction, -1.0, 1.0)

            losses = self.loss_fn(corrected, cloudy)
            losses = {k: torch.nan_to_num(v, nan=0.0, posinf=1.0, neginf=0.0) if isinstance(v, torch.Tensor) else v for k, v in losses.items()}

            self.optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            for k, v in losses.items():
                total_losses[k] = total_losses.get(k, 0) + v.item() * images.size(0)

        n = len(loader.dataset)
        if n == 0:
            return {"total": 0.0, "l1": 0.0, "ssim": 0.0, "spectral": 0.0}
        return {k: v / n for k, v in total_losses.items()}

    @torch.no_grad()
    def validate(self, loader: DataLoader, density_model: nn.Module = None) -> dict:
        self.model.eval()
        total_losses = {}

        for images, density_targets in tqdm(loader, desc="Validating", leave=False):
            images = images.to(self.device)
            if images.shape[1] == 1:
                images = images.repeat(1, 3, 1, 1)

            cloudy = images
            density = density_targets.to(self.device) if density_targets is not None else None
            if density is None and density_model is not None:
                density = density_model(cloudy)

            correction = self.model(cloudy, density if density is not None else torch.zeros_like(cloudy[:, :1]))
            corrected = cloudy + torch.clamp(correction, -1.0, 1.0)
            losses = self.loss_fn(corrected, cloudy)
            losses = {k: torch.nan_to_num(v, nan=0.0, posinf=1.0, neginf=0.0) if isinstance(v, torch.Tensor) else v for k, v in losses.items()}

            for k, v in losses.items():
                total_losses[k] = total_losses.get(k, 0) + v.item() * images.size(0)

        n = len(loader.dataset)
        if n == 0:
            return {"total": 0.0, "l1": 0.0, "ssim": 0.0, "spectral": 0.0}
        return {k: v / n for k, v in total_losses.items()}

    def fit(self, train_loader: DataLoader, val_loader: DataLoader,
            epochs: int = 50, checkpoint_dir: Path = None,
            density_model: nn.Module = None) -> dict:
        checkpoint_dir = Path(checkpoint_dir or CHECKPOINTS / "correction_model")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        best_val_loss = float("inf")
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(1, epochs + 1):
            train_metrics = self.train_epoch(train_loader, density_model)
            val_metrics = self.validate(val_loader, density_model)
            self.scheduler.step(val_metrics["total"])

            history["train_loss"].append(train_metrics["total"])
            history["val_loss"].append(val_metrics["total"])

            print(f"Epoch {epoch:3d}/{epochs}  "
                  f"Train: {train_metrics['total']:.6f}  "
                  f"Val: {val_metrics['total']:.6f}  "
                  f"L1: {val_metrics.get('l1', 0):.6f}  "
                  f"SSIM: {val_metrics.get('ssim', 0):.6f}")

            if val_metrics["total"] < best_val_loss:
                best_val_loss = val_metrics["total"]
                torch.save(self.model.state_dict(), checkpoint_dir / "best_model.pth")
                print(f"  -> Saved best model (val_loss={best_val_loss:.6f})")

        torch.save(self.model.state_dict(), checkpoint_dir / "final_model.pth")
        return history


if __name__ == "__main__":
    from src.models.thin_cloud_correction import ThinCloudCorrection

    model = ThinCloudCorrection(in_channels=3)
    trainer = CorrectionTrainer(model)

    patch_dir = PATCHES
    if not patch_dir.exists() or not any(patch_dir.rglob("*.npy")):
        from src.training.synthetic_data import generate_synthetic_patches
        patch_dir = generate_synthetic_patches()

    train_loader, val_loader, _ = create_dataloaders(patch_dir, batch_size=8)
    trainer.fit(train_loader, val_loader, epochs=10)
