import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional, Union
from torch.utils.data import Dataset, DataLoader, IterableDataset
from tqdm import tqdm

from src.config import PATCHES, CHECKPOINTS, RANDOM_SEED, SEN12MS_RAW
from src.training.losses import FilteredJaccardLoss


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return nn.functional.mse_loss(pred, target)


def mae_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return nn.functional.l1_loss(pred, target)


class PatchDataset(Dataset):
    def __init__(self, split: str, patch_dir: Path = None, transform=None):
        self.patch_dir = Path(patch_dir or PATCHES) / split
        self.transform = transform
        self.files = sorted(self.patch_dir.glob("*.npy"))
        self.metas = {}
        for f in self.files:
            meta_path = f.parent / f"{f.stem}_meta.json"
            if meta_path.exists():
                self.metas[f.stem] = json.loads(meta_path.read_text())

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        npy_path = self.files[idx]
        stem = npy_path.stem

        image = np.load(str(npy_path)).astype(np.float32)
        image = image / 1023.0

        meta = self.metas.get(stem, {})
        cloud_frac = meta.get("cloud_fraction", 0.0)
        density_map = np.full((image.shape[0], image.shape[1]), cloud_frac, dtype=np.float32)

        tensor = torch.from_numpy(image).permute(2, 0, 1)
        target = torch.from_numpy(density_map).unsqueeze(0)

        return tensor, target


class DensityTrainer:
    """Trainer for CloudDensityNet with AMP and Colab checkpoint resume hooks."""

    def __init__(
        self,
        model: nn.Module,
        device: str = None,
        use_amp: bool = False,
        use_fjl: bool = False,
        fjl_gl_type: str = "GL1",
    ):
        self.model = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.use_amp = use_amp and ("cuda" in self.device)
        device_type = "cuda" if "cuda" in self.device else "cpu"
        self.scaler = torch.amp.GradScaler(device_type, enabled=self.use_amp)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )
        self.use_fjl = use_fjl
        self.fjl_loss = FilteredJaccardLoss(gl_type=fjl_gl_type) if use_fjl else None

    def compute_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.use_fjl and self.fjl_loss is not None:
            return 0.5 * mse_loss(outputs, targets) + 0.5 * self.fjl_loss(outputs, targets)
        return mse_loss(outputs, targets)

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        device_type = "cuda" if "cuda" in self.device else "cpu"

        for batch in tqdm(loader, desc="Training", leave=False):
            if isinstance(batch, (list, tuple)):
                images, targets = batch[0], batch[1]
                dem = batch[3] if len(batch) > 3 else None
            elif isinstance(batch, dict):
                images = batch["s2_cloudy"]
                targets = batch.get("cloud_mask", images.mean(dim=1, keepdim=True))
                dem = batch.get("dem", None)
            else:
                continue

            images = images.to(self.device)
            targets = targets.to(self.device)
            if dem is not None:
                dem = dem.to(self.device)

            self.optimizer.zero_grad()

            with torch.amp.autocast(device_type, enabled=self.use_amp):
                outputs = self.model(images, dem=dem) if dem is not None else self.model(images)
                loss = self.compute_loss(outputs, targets)

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item() * images.size(0)

        n = len(loader.dataset)
        return total_loss / n if n > 0 else 0.0

    @torch.no_grad()
    def validate(self, loader: DataLoader) -> dict:
        self.model.eval()
        total_mse = 0.0
        total_mae = 0.0
        device_type = "cuda" if "cuda" in self.device else "cpu"

        for batch in tqdm(loader, desc="Validating", leave=False):
            if isinstance(batch, (list, tuple)):
                images, targets = batch[0], batch[1]
                dem = batch[3] if len(batch) > 3 else None
            elif isinstance(batch, dict):
                images = batch["s2_cloudy"]
                targets = batch.get("cloud_mask", images.mean(dim=1, keepdim=True))
                dem = batch.get("dem", None)
            else:
                continue

            images = images.to(self.device)
            targets = targets.to(self.device)
            if dem is not None:
                dem = dem.to(self.device)

            with torch.amp.autocast(device_type, enabled=self.use_amp):
                outputs = self.model(images, dem=dem) if dem is not None else self.model(images)
                total_mse += mse_loss(outputs, targets).item() * images.size(0)
                total_mae += mae_loss(outputs, targets).item() * images.size(0)

        n = len(loader.dataset)
        if n == 0:
            return {"mse": 0.0, "mae": 0.0, "rmse": 0.0}
        return {"mse": total_mse / n, "mae": total_mae / n, "rmse": float(np.sqrt(total_mse / n))}

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
        checkpoint_dir: Path = None,
        resume_from: Optional[Union[str, Path]] = None,
    ) -> dict:
        checkpoint_dir = Path(checkpoint_dir or CHECKPOINTS / "density_model")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        start_epoch = 1
        best_val_loss = float("inf")
        history = {"train_loss": [], "val_mse": [], "val_mae": [], "val_rmse": []}

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
            val_metrics = self.validate(val_loader)
            self.scheduler.step(val_metrics["mse"])

            history["train_loss"].append(train_loss)
            history["val_mse"].append(val_metrics["mse"])
            history["val_mae"].append(val_metrics["mae"])
            history["val_rmse"].append(val_metrics["rmse"])

            print(
                f"Epoch {epoch:3d}/{epochs}  "
                f"Train: {train_loss:.6f}  "
                f"Val MSE: {val_metrics['mse']:.6f}  "
                f"MAE: {val_metrics['mae']:.6f}  "
                f"RMSE: {val_metrics['rmse']:.6f}"
            )

            state = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "history": history,
            }
            torch.save(state, checkpoint_dir / "latest_checkpoint.pth")

            if val_metrics["mse"] < best_val_loss:
                best_val_loss = val_metrics["mse"]
                torch.save(self.model.state_dict(), checkpoint_dir / "best_model.pth")
                print(f"  -> Saved best model (val_loss={best_val_loss:.6f})")

        torch.save(self.model.state_dict(), checkpoint_dir / "final_model.pth")
        with open(checkpoint_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)

        return history

    def load_checkpoint(self, path: Path):
        ckpt = torch.load(path, map_location="cpu")
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        else:
            state = ckpt
        device = next(self.model.parameters()).device
        self.model.load_state_dict({k: v.to(device) for k, v in state.items()})
        self.model.eval()


def create_dataloaders(
    patch_dir: Path = None,
    batch_size: int = 16,
    num_workers: int = 0,
    dataset_type: str = "auto",
) -> tuple[DataLoader, DataLoader, DataLoader]:
    patch_path = Path(patch_dir or PATCHES)

    if dataset_type == "sen12ms_cr_streaming":
        from src.data.sen12ms_dataset import SEN12MSCRStreamingIterable
        train_ds = SEN12MSCRStreamingIterable(split="train", return_dict=True, shuffle_buffer=10000)
        val_ds = SEN12MSCRStreamingIterable(split="validation", return_dict=True, shuffle_buffer=1000)
        test_ds = SEN12MSCRStreamingIterable(split="test", return_dict=True, shuffle_buffer=1000)
        train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers)
        val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers)
        test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers)
    elif dataset_type == "auto" and patch_path.exists() and any(patch_path.rglob("*.npy")):
        # Auto-detect patch format (.npy files)
        train_ds = PatchDataset("train", patch_path)
        val_ds = PatchDataset("val", patch_path)
        test_ds = PatchDataset("test", patch_path)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    elif dataset_type == "sen12ms_cr" or (dataset_type == "auto" and (patch_dir and Path(patch_dir).exists() and any(Path(patch_dir).rglob("*.tif*")))):
        from src.data.sen12ms_dataset import SEN12MSCRDataset
        data_root = patch_dir
        train_ds = SEN12MSCRDataset(data_root, split="train", return_dict=True)
        val_ds = SEN12MSCRDataset(data_root, split="val", return_dict=True)
        test_ds = SEN12MSCRDataset(data_root, split="test", return_dict=True)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    elif dataset_type == "sen12ms_cr" or (dataset_type == "auto" and (SEN12MS_RAW.exists() and any(SEN12MS_RAW.rglob("*.tif*")))):
        from src.data.sen12ms_dataset import SEN12MSCRDataset
        data_root = SEN12MS_RAW
        train_ds = SEN12MSCRDataset(data_root, split="train", return_dict=True)
        val_ds = SEN12MSCRDataset(data_root, split="val", return_dict=True)
        test_ds = SEN12MSCRDataset(data_root, split="test", return_dict=True)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    else:
        train_ds = PatchDataset("train", patch_path)
        val_ds = PatchDataset("val", patch_path)
        test_ds = PatchDataset("test", patch_path)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    from src.models.cloud_density import CloudDensityNet

    model = CloudDensityNet(in_channels=3, out_channels=1)
    trainer = DensityTrainer(model)

    patch_dir = PATCHES
    if not patch_dir.exists() or not any(patch_dir.rglob("*.npy")):
        print(f"[WARN] No patches found in {patch_dir}. Generating synthetic data fixture for testing.")
        from src.training.synthetic_data import generate_synthetic_patches
        patch_dir = generate_synthetic_patches()

    train_loader, val_loader, test_loader = create_dataloaders(patch_dir, batch_size=8)
    trainer.fit(train_loader, val_loader, epochs=10)
