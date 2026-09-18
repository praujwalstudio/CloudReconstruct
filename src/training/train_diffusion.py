import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional, Union
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import PATCHES, CHECKPOINTS
from src.training.train_density import PatchDataset, create_dataloaders


def cosine_beta_schedule(steps: int, s: float = 0.008) -> torch.Tensor:
    t = torch.linspace(0, steps, steps + 1, dtype=torch.float64)
    f_t = torch.cos((t / steps + s) / (1 + s) * np.pi / 2) ** 2
    alphas_cumprod = f_t / f_t[0]
    betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    return torch.clamp(betas, max=0.999)


class DiffusionSchedule:
    def __init__(self, noise_steps: int = 100, schedule: str = "cosine"):
        self.noise_steps = noise_steps
        if schedule == "cosine":
            betas = cosine_beta_schedule(noise_steps)
        else:
            betas = torch.linspace(1e-4, 0.02, noise_steps)
        self.betas = betas.float()
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor,
                 noise: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_ab = self.alpha_bars[t].sqrt().view(-1, 1, 1, 1)
        sqrt_1m_ab = (1 - self.alpha_bars[t]).sqrt().view(-1, 1, 1, 1)
        x_t = sqrt_ab * x_0 + sqrt_1m_ab * noise
        return x_t, noise

    def p_sample(self, model: nn.Module, x_t: torch.Tensor, t: torch.Tensor,
                 condition: torch.Tensor, dem: torch.Tensor = None) -> torch.Tensor:
        t_batch = t.expand(x_t.shape[0])
        predicted_noise = model(x_t, condition, t_batch, dem=dem) if dem is not None else model(x_t, condition, t_batch)
        alpha = self.alphas[t].view(-1, 1, 1, 1)
        alpha_bar = self.alpha_bars[t].view(-1, 1, 1, 1)
        beta = self.betas[t].view(-1, 1, 1, 1)

        x_0_pred = (x_t - beta * predicted_noise / (1 - alpha_bar).sqrt()) / alpha.sqrt()

        if t[0].item() > 0:
            noise = torch.randn_like(x_t)
            posterior_var = beta * (1 - alpha_bar / self.alpha_bars[t - 1].view(-1, 1, 1, 1)) / (1 - alpha_bar)
            x_prev = x_0_pred + posterior_var.sqrt() * noise
        else:
            x_prev = x_0_pred

        return x_prev

    @torch.no_grad()
    def sample(self, model: nn.Module, shape: tuple, condition: torch.Tensor,
               device: str = "cpu", dem: torch.Tensor = None) -> torch.Tensor:
        x_t = torch.randn(shape, device=device)
        for i in reversed(range(self.noise_steps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            x_t = self.p_sample(model, x_t, t, condition, dem=dem)
        return x_t


class DiffusionTrainer:
    """Trainer for SAR Conditional Diffusion with AMP and Colab checkpoint recovery."""

    def __init__(
        self,
        model: nn.Module,
        schedule: DiffusionSchedule = None,
        device: str = None,
        use_amp: bool = False,
    ):
        self.model = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.use_amp = use_amp and ("cuda" in self.device)
        device_type = "cuda" if "cuda" in self.device else "cpu"
        self.scaler = torch.amp.GradScaler(device_type, enabled=self.use_amp)
        self.schedule = schedule or DiffusionSchedule(noise_steps=getattr(model, "noise_steps", 100))
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)

    def _compute_loss(
        self,
        clear: torch.Tensor,
        cloudy: torch.Tensor = None,
        sar: torch.Tensor = None,
        dem: torch.Tensor = None,
        cloud_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        b = clear.shape[0]
        if cloudy is None:
            cloudy = clear
        if sar is None:
            sar = torch.zeros(b, 2, *clear.shape[2:], device=self.device)

        if hasattr(self.model, "compute_loss"):
            return self.model.compute_loss(clear, cloudy, sar, dem=dem, cloud_mask=cloud_mask)

        t = torch.randint(0, self.schedule.noise_steps, (b,), device=self.device)
        noise = torch.randn_like(clear)
        x_t, _ = self.schedule.q_sample(clear, t, noise)
        predicted_noise = self.model(x_t, cloudy, t)
        return self.loss_fn(predicted_noise, noise)

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        device_type = "cuda" if "cuda" in self.device else "cpu"

        for batch in tqdm(loader, desc="Training", leave=False):
            sar, dem, mask = None, None, None
            if isinstance(batch, (list, tuple)):
                clear = batch[0]
                cloudy = batch[0]
                if len(batch) > 1 and batch[1] is not None:
                    # Could be mask or cloudy target
                    if batch[1].shape[1] == 1:
                        mask = batch[1]
                    else:
                        cloudy = batch[1]
                if len(batch) > 2:
                    sar = batch[2]
                if len(batch) > 3:
                    dem = batch[3]
            elif isinstance(batch, dict):
                clear = batch.get("s2_clear", batch.get("s2_cloudy"))
                cloudy = batch.get("s2_cloudy", clear)
                sar = batch.get("s1", None)
                dem = batch.get("dem", None)
                mask = batch.get("cloud_mask", None)
            else:
                continue

            clear = clear.to(self.device)
            cloudy = cloudy.to(self.device)
            if sar is not None:
                sar = sar.to(self.device)
            if dem is not None:
                dem = dem.to(self.device)
            if mask is not None:
                mask = mask.to(self.device)

            if clear.shape[1] == 1:
                clear = clear.repeat(1, 3, 1, 1)
            if cloudy.shape[1] == 1:
                cloudy = cloudy.repeat(1, 3, 1, 1)

            self.optimizer.zero_grad()

            with torch.amp.autocast(device_type, enabled=self.use_amp):
                loss = self._compute_loss(clear, cloudy=cloudy, sar=sar, dem=dem, cloud_mask=mask)

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            total_loss += loss.item() * clear.size(0)

        n = len(loader.dataset)
        return total_loss / n if n > 0 else 0.0

    @torch.no_grad()
    def validate(self, loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        device_type = "cuda" if "cuda" in self.device else "cpu"

        for batch in tqdm(loader, desc="Validating", leave=False):
            sar, dem, mask = None, None, None
            if isinstance(batch, (list, tuple)):
                clear = batch[0]
                cloudy = batch[0]
                if len(batch) > 1 and batch[1] is not None:
                    if batch[1].shape[1] == 1:
                        mask = batch[1]
                    else:
                        cloudy = batch[1]
                if len(batch) > 2:
                    sar = batch[2]
                if len(batch) > 3:
                    dem = batch[3]
            elif isinstance(batch, dict):
                clear = batch.get("s2_clear", batch.get("s2_cloudy"))
                cloudy = batch.get("s2_cloudy", clear)
                sar = batch.get("s1", None)
                dem = batch.get("dem", None)
                mask = batch.get("cloud_mask", None)
            else:
                continue

            clear = clear.to(self.device)
            cloudy = cloudy.to(self.device)
            if sar is not None:
                sar = sar.to(self.device)
            if dem is not None:
                dem = dem.to(self.device)
            if mask is not None:
                mask = mask.to(self.device)

            if clear.shape[1] == 1:
                clear = clear.repeat(1, 3, 1, 1)
            if cloudy.shape[1] == 1:
                cloudy = cloudy.repeat(1, 3, 1, 1)

            with torch.amp.autocast(device_type, enabled=self.use_amp):
                loss = self._compute_loss(clear, cloudy=cloudy, sar=sar, dem=dem, cloud_mask=mask)
            total_loss += loss.item() * clear.size(0)

        n = len(loader.dataset)
        return total_loss / n if n > 0 else 0.0

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        checkpoint_dir: Path = None,
        resume_from: Optional[Union[str, Path]] = None,
    ) -> dict:
        checkpoint_dir = Path(checkpoint_dir or CHECKPOINTS / "diffusion_model")
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
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            self.scheduler.step()

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            print(
                f"Epoch {epoch:3d}/{epochs}  "
                f"Train: {train_loss:.6f}  "
                f"Val: {val_loss:.6f}"
            )

            state = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "history": history,
            }
            torch.save(state, checkpoint_dir / "latest_checkpoint.pth")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), checkpoint_dir / "best_model.pth")
                print(f"  -> Saved best model (val_loss={best_val_loss:.6f})")

            if epoch % 10 == 0:
                torch.save(self.model.state_dict(), checkpoint_dir / f"model_epoch_{epoch}.pth")

        torch.save(self.model.state_dict(), checkpoint_dir / "final_model.pth")
        return history


if __name__ == "__main__":
    from src.models.sar_fusion import SARDiffusionWrapper

    model = SARDiffusionWrapper(sar_channels=2, liss4_channels=3, out_channels=3)
    trainer = DiffusionTrainer(model)

    patch_dir = PATCHES
    if not patch_dir.exists() or not any(patch_dir.rglob("*.npy")):
        from src.training.synthetic_data import generate_synthetic_patches
        patch_dir = generate_synthetic_patches()

    train_loader, val_loader, _ = create_dataloaders(patch_dir, batch_size=8)
    trainer.fit(train_loader, val_loader, epochs=10)
