import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Union
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import PATCHES, CHECKPOINTS
from src.training.train_density import PatchDataset, create_dataloaders
from src.training.losses import CombinedLoss, MultiScaleConvergenceLoss, HeteroscedasticNLLLoss


class TemporalPairDataset(PatchDataset):
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cloudy, density = super().__getitem__(idx)
        ref_idx = (idx + 1) % len(self.files)
        ref_path = self.files[ref_idx]
        ref = np.load(str(ref_path)).astype(np.float32) / 1023.0
        ref_tensor = torch.from_numpy(ref).permute(2, 0, 1)
        return cloudy, ref_tensor, cloudy, density

    def __init__(self, split: str, patch_dir: Path = None, transform=None):
        super().__init__(split, patch_dir, transform)


class TemporalTrainer:
    """Trainer for TemporalFusion with AMP, Heteroscedastic NLL Loss, and Colab resume hooks."""

    def __init__(
        self,
        model: nn.Module,
        device: str = None,
        loss_weights: dict = None,
        use_amp: bool = False,
        use_multiscale_loss: bool = True,
        use_uncertainty: bool = False,
    ):
        self.model = model
        self.use_uncertainty = use_uncertainty
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.use_amp = use_amp and ("cuda" in self.device)
        device_type = "cuda" if "cuda" in self.device else "cpu"
        self.scaler = torch.amp.GradScaler(device_type, enabled=self.use_amp)

        self.ms_loss = MultiScaleConvergenceLoss() if use_multiscale_loss else None
        self.nll_loss = HeteroscedasticNLLLoss()
        self.combined_loss = CombinedLoss(loss_weights or {}, device=self.device)

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )

    def _compute_loss(self, output: Union[torch.Tensor, tuple], target: torch.Tensor, density: torch.Tensor) -> dict:
        losses = {}
        if isinstance(output, (tuple, list)):
            pred, log_var = output[0], output[1]
            nll = self.nll_loss(pred, target, log_var)
            losses["nll"] = nll
            if self.ms_loss is not None:
                ms = self.ms_loss(pred, target)
                losses["multiscale"] = ms
                losses["total"] = 0.5 * nll + 0.5 * ms
            else:
                c_loss = self.combined_loss(pred, target, cloud_mask=density)
                losses.update(c_loss)
                losses["total"] = 0.5 * nll + 0.5 * c_loss["total"]
        else:
            pred = output
            if self.ms_loss is not None:
                ms = self.ms_loss(pred, target)
                losses["total"] = ms
                losses["multiscale"] = ms
            else:
                losses = self.combined_loss(pred, target, cloud_mask=density)
        return losses

    def train_epoch(self, loader: DataLoader) -> dict:
        self.model.train()
        total_losses = {}
        device_type = "cuda" if "cuda" in self.device else "cpu"

        for cloudy, ref, target, density in tqdm(loader, desc="Training", leave=False):
            cloudy = cloudy.to(self.device)
            ref = ref.to(self.device)
            target = target.to(self.device)
            density = density.to(self.device)

            if cloudy.shape[1] == 1:
                cloudy = cloudy.repeat(1, 3, 1, 1)
                ref = ref.repeat(1, 3, 1, 1)
                target = target.repeat(1, 3, 1, 1)

            self.optimizer.zero_grad()

            with torch.amp.autocast(device_type, enabled=self.use_amp):
                if self.use_uncertainty and hasattr(self.model, "forward") and "return_uncertainty" in self.model.forward.__code__.co_varnames:
                    output = self.model(cloudy, ref, density, return_uncertainty=True)
                else:
                    output = self.model(cloudy, ref, density)
                losses = self._compute_loss(output, target, density)

            if self.use_amp:
                self.scaler.scale(losses["total"]).backward()
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                if not (torch.isnan(grad_norm) or torch.isinf(grad_norm)):
                    self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                losses["total"].backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                if not (torch.isnan(grad_norm) or torch.isinf(grad_norm)):
                    self.optimizer.step()

            for k, v in losses.items():
                val = v.item() if isinstance(v, torch.Tensor) else v
                total_losses[k] = total_losses.get(k, 0) + val * cloudy.size(0)

        n = len(loader.dataset)
        if n == 0:
            return {"total": 0.0, "l1": 0.0, "ssim": 0.0, "spectral": 0.0}
        return {k: v / n for k, v in total_losses.items()}

    @torch.no_grad()
    def validate(self, loader: DataLoader) -> dict:
        self.model.eval()
        total_losses = {}
        device_type = "cuda" if "cuda" in self.device else "cpu"

        for cloudy, ref, target, density in tqdm(loader, desc="Validating", leave=False):
            cloudy = cloudy.to(self.device)
            ref = ref.to(self.device)
            target = target.to(self.device)
            density = density.to(self.device)

            if cloudy.shape[1] == 1:
                cloudy = cloudy.repeat(1, 3, 1, 1)
                ref = ref.repeat(1, 3, 1, 1)
                target = target.repeat(1, 3, 1, 1)

            with torch.amp.autocast(device_type, enabled=self.use_amp):
                if self.use_uncertainty and hasattr(self.model, "forward") and "return_uncertainty" in self.model.forward.__code__.co_varnames:
                    output = self.model(cloudy, ref, density, return_uncertainty=True)
                else:
                    output = self.model(cloudy, ref, density)
                losses = self._compute_loss(output, target, density)

            for k, v in losses.items():
                val = v.item() if isinstance(v, torch.Tensor) else v
                total_losses[k] = total_losses.get(k, 0) + val * cloudy.size(0)

        n = len(loader.dataset)
        if n == 0:
            return {"total": 0.0, "l1": 0.0, "ssim": 0.0, "spectral": 0.0}
        return {k: v / n for k, v in total_losses.items()}

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
        checkpoint_dir: Path = None,
        resume_from: Optional[Union[str, Path]] = None,
    ) -> dict:
        checkpoint_dir = Path(checkpoint_dir or CHECKPOINTS / "temporal_model")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        start_epoch = 1
        best_val_loss = float("inf")
        history = {"train_loss": [], "val_loss": []}

        if resume_from and Path(resume_from).exists():
            print(f"[RESUME] Loading checkpoint from {resume_from}")
            ckpt = torch.load(resume_from, map_location=self.device)
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                self.model.load_state_dict(ckpt["model_state_dict"])
                self.optimizer.load_state_dict(ckpt.get("optimizer_state_dict", self.optimizer.state_dict()))
                start_epoch = ckpt.get("epoch", 0) + 1
                best_val_loss = ckpt.get("best_val_loss", float("inf"))
                history = ckpt.get("history", history)
            else:
                self.model.load_state_dict(ckpt)

        for epoch in range(start_epoch, epochs + 1):
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)
            self.scheduler.step(val_metrics["total"])

            history["train_loss"].append(train_metrics["total"])
            history["val_loss"].append(val_metrics["total"])

            print(
                f"Epoch {epoch:3d}/{epochs}  "
                f"Train: {train_metrics['total']:.6f}  "
                f"Val: {val_metrics['total']:.6f}  "
                f"L1: {val_metrics.get('l1', 0):.6f}  "
                f"SSIM: {val_metrics.get('ssim', 0):.6f}"
            )

            state = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "history": history,
            }
            torch.save(state, checkpoint_dir / "latest_checkpoint.pth")

            if val_metrics["total"] < best_val_loss:
                best_val_loss = val_metrics["total"]
                torch.save(self.model.state_dict(), checkpoint_dir / "best_model.pth")
                print(f"  -> Saved best model (val_loss={best_val_loss:.6f})")

        torch.save(self.model.state_dict(), checkpoint_dir / "final_model.pth")
        return history


def create_temporal_dataloaders(
    patch_dir: Path = None, batch_size: int = 8, num_workers: int = 0
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = TemporalPairDataset("train", patch_dir)
    val_ds = TemporalPairDataset("val", patch_dir)
    test_ds = TemporalPairDataset("test", patch_dir)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    from src.models.temporal_fusion import TemporalFusion

    model = TemporalFusion(in_channels=3)
    trainer = TemporalTrainer(model)

    patch_dir = PATCHES
    if not patch_dir.exists() or not any(patch_dir.rglob("*.npy")):
        from src.training.synthetic_data import generate_synthetic_patches
        patch_dir = generate_synthetic_patches()

    train_loader, val_loader, _ = create_temporal_dataloaders(patch_dir, batch_size=4)
    trainer.fit(train_loader, val_loader, epochs=10)
