import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Union
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import PATCHES, CHECKPOINTS
from src.training.train_density import PatchDataset, create_dataloaders
from src.training.losses import CombinedLoss, SpatialAttentionLoss, LSGANLoss, CARLLoss


class CorrectionTrainer:
    """Trainer for ThinCloudCorrection and SpA-GAN with AMP, LSGAN, and checkpoint hooks."""

    def __init__(
        self,
        model: nn.Module,
        discriminator: Optional[nn.Module] = None,
        device: str = None,
        loss_weights: dict = None,
        use_amp: bool = False,
    ):
        self.model = model
        self.discriminator = discriminator
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        if self.discriminator is not None:
            self.discriminator.to(self.device)

        self.use_amp = use_amp and ("cuda" in self.device)
        device_type = "cuda" if "cuda" in self.device else "cpu"
        self.scaler = torch.amp.GradScaler(device_type, enabled=self.use_amp)

        self.loss_weights = loss_weights or {"l1": 1.0, "ssim": 0.5, "perceptual": 0.0, "att": 0.5, "adv": 0.01}
        self.loss_fn = CombinedLoss(self.loss_weights, device=self.device)
        self.att_loss = SpatialAttentionLoss()
        self.lsgan = LSGANLoss()

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
        if self.discriminator is not None:
            self.opt_d = torch.optim.AdamW(self.discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
        else:
            self.opt_d = None

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )

    def train_epoch(self, loader: DataLoader, density_model: nn.Module = None) -> dict:
        self.model.train()
        if self.discriminator is not None:
            self.discriminator.train()

        total_losses = {}
        device_type = "cuda" if "cuda" in self.device else "cpu"

        for batch in tqdm(loader, desc="Training", leave=False):
            if isinstance(batch, (list, tuple)):
                cloudy = batch[0]
                target = batch[2] if len(batch) > 2 else batch[0]
                density_targets = batch[1] if len(batch) > 1 else None
            elif isinstance(batch, dict):
                cloudy = batch["s2_cloudy"]
                target = batch.get("s2_clear", batch["s2_cloudy"])
                density_targets = batch.get("cloud_mask", None)
            else:
                continue

            cloudy = cloudy.to(self.device)
            target = target.to(self.device)
            if cloudy.shape[1] == 1:
                cloudy = cloudy.repeat(1, 3, 1, 1)
            if target.shape[1] == 1:
                target = target.repeat(1, 3, 1, 1)

            density = density_targets.to(self.device) if density_targets is not None else None
            if density is None and density_model is not None:
                with torch.no_grad():
                    density = density_model(cloudy)
            if density is None:
                density = torch.zeros((cloudy.shape[0], 1, cloudy.shape[2], cloudy.shape[3]), device=self.device)

            # ----------------------------------------------------
            # 1. Forward Generator
            # ----------------------------------------------------
            self.optimizer.zero_grad()

            with torch.amp.autocast(device_type, enabled=self.use_amp):
                if hasattr(self.model, "forward") and "return_attn" in self.model.forward.__code__.co_varnames:
                    corrected, attn = self.model(cloudy, density, return_attn=True)
                else:
                    corrected = self.model(cloudy, density)
                    attn = None

                corrected = torch.clamp(corrected, 0.0, 1.0)
                losses = self.loss_fn(corrected, target, cloud_mask=density)

                if attn is not None and self.loss_weights.get("att", 0) > 0:
                    l_att = self.att_loss(corrected, target, attn)
                    losses["att"] = l_att
                    losses["total"] = losses["total"] + self.loss_weights.get("att", 0.5) * l_att

                if self.discriminator is not None and self.loss_weights.get("adv", 0) > 0:
                    d_fake = self.discriminator(corrected)
                    l_adv = self.lsgan.generator_loss(d_fake)
                    losses["adv"] = l_adv
                    losses["total"] = losses["total"] + self.loss_weights.get("adv", 0.01) * l_adv

            if self.use_amp:
                self.scaler.scale(losses["total"]).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            # ----------------------------------------------------
            # 2. Update Discriminator (if enabled)
            # ----------------------------------------------------
            if self.discriminator is not None and self.opt_d is not None:
                self.opt_d.zero_grad()
                with torch.amp.autocast(device_type, enabled=self.use_amp):
                    d_real = self.discriminator(target)
                    d_fake_det = self.discriminator(corrected.detach())
                    loss_d = self.lsgan.discriminator_loss(d_real, d_fake_det)
                    losses["loss_d"] = loss_d

                if self.use_amp:
                    self.scaler.scale(loss_d).backward()
                    self.scaler.unscale_(self.opt_d)
                    torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
                    self.scaler.step(self.opt_d)
                    self.scaler.update()
                else:
                    loss_d.backward()
                    torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
                    self.opt_d.step()

            for k, v in losses.items():
                val = v.item() if isinstance(v, torch.Tensor) else v
                total_losses[k] = total_losses.get(k, 0) + val * cloudy.size(0)

        n = len(loader.dataset)
        if n == 0:
            return {"total": 0.0, "l1": 0.0, "ssim": 0.0}
        return {k: v / n for k, v in total_losses.items()}

    @torch.no_grad()
    def validate(self, loader: DataLoader, density_model: nn.Module = None) -> dict:
        self.model.eval()
        total_losses = {}
        device_type = "cuda" if "cuda" in self.device else "cpu"

        for batch in tqdm(loader, desc="Validating", leave=False):
            if isinstance(batch, (list, tuple)):
                cloudy = batch[0]
                target = batch[2] if len(batch) > 2 else batch[0]
                density_targets = batch[1] if len(batch) > 1 else None
            elif isinstance(batch, dict):
                cloudy = batch["s2_cloudy"]
                target = batch.get("s2_clear", batch["s2_cloudy"])
                density_targets = batch.get("cloud_mask", None)
            else:
                continue

            cloudy = cloudy.to(self.device)
            target = target.to(self.device)
            if cloudy.shape[1] == 1:
                cloudy = cloudy.repeat(1, 3, 1, 1)
            if target.shape[1] == 1:
                target = target.repeat(1, 3, 1, 1)

            density = density_targets.to(self.device) if density_targets is not None else None
            if density is None and density_model is not None:
                density = density_model(cloudy)
            if density is None:
                density = torch.zeros((cloudy.shape[0], 1, cloudy.shape[2], cloudy.shape[3]), device=self.device)

            with torch.amp.autocast(device_type, enabled=self.use_amp):
                if hasattr(self.model, "forward") and "return_attn" in self.model.forward.__code__.co_varnames:
                    corrected, _ = self.model(cloudy, density, return_attn=True)
                else:
                    corrected = self.model(cloudy, density)
                corrected = torch.clamp(corrected, 0.0, 1.0)
                losses = self.loss_fn(corrected, target, cloud_mask=density)

            for k, v in losses.items():
                val = v.item() if isinstance(v, torch.Tensor) else v
                total_losses[k] = total_losses.get(k, 0) + val * cloudy.size(0)

        n = len(loader.dataset)
        if n == 0:
            return {"total": 0.0, "l1": 0.0, "ssim": 0.0}
        return {k: v / n for k, v in total_losses.items()}

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
        checkpoint_dir: Path = None,
        density_model: nn.Module = None,
        resume_from: Optional[Union[str, Path]] = None,
    ) -> dict:
        checkpoint_dir = Path(checkpoint_dir or CHECKPOINTS / "correction_model")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        start_epoch = 1
        best_val_loss = float("inf")
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(start_epoch, epochs + 1):
            train_metrics = self.train_epoch(train_loader, density_model)
            val_metrics = self.validate(val_loader, density_model)
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
