import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.config import PATCHES, CHECKPOINTS, RANDOM_SEED


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
    def __init__(self, model: nn.Module, device: str = None):
        self.model = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0

        for images, targets in tqdm(loader, desc="Training", leave=False):
            images = images.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = mse_loss(outputs, targets)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * images.size(0)

        return total_loss / len(loader.dataset)

    @torch.no_grad()
    def validate(self, loader: DataLoader) -> dict:
        self.model.eval()
        total_mse = 0.0
        total_mae = 0.0

        for images, targets in tqdm(loader, desc="Validating", leave=False):
            images = images.to(self.device)
            targets = targets.to(self.device)

            outputs = self.model(images)
            total_mse += mse_loss(outputs, targets).item() * images.size(0)
            total_mae += mae_loss(outputs, targets).item() * images.size(0)

        n = len(loader.dataset)
        if n == 0:
            return {"mse": 0.0, "mae": 0.0, "rmse": 0.0}
        return {"mse": total_mse / n, "mae": total_mae / n, "rmse": np.sqrt(total_mse / n)}

    def fit(self, train_loader: DataLoader, val_loader: DataLoader,
            epochs: int = 50, checkpoint_dir: Path = None) -> dict:
        checkpoint_dir = Path(checkpoint_dir or CHECKPOINTS / "density_model")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        best_val_loss = float("inf")
        history = {"train_loss": [], "val_mse": [], "val_mae": [], "val_rmse": []}

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)
            self.scheduler.step(val_metrics["mse"])

            history["train_loss"].append(train_loss)
            history["val_mse"].append(val_metrics["mse"])
            history["val_mae"].append(val_metrics["mae"])
            history["val_rmse"].append(val_metrics["rmse"])

            print(f"Epoch {epoch:3d}/{epochs}  "
                  f"Train: {train_loss:.6f}  "
                  f"Val MSE: {val_metrics['mse']:.6f}  "
                  f"MAE: {val_metrics['mae']:.6f}  "
                  f"RMSE: {val_metrics['rmse']:.6f}")

            if val_metrics["mse"] < best_val_loss:
                best_val_loss = val_metrics["mse"]
                torch.save(self.model.state_dict(), checkpoint_dir / "best_model.pth")
                print(f"  -> Saved best model (val_loss={best_val_loss:.6f})")

        torch.save(self.model.state_dict(), checkpoint_dir / "final_model.pth")
        with open(checkpoint_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)

        return history

    def load_checkpoint(self, path: Path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()


def create_dataloaders(patch_dir: Path = None, batch_size: int = 16,
                       num_workers: int = 0) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = PatchDataset("train", patch_dir)
    val_ds = PatchDataset("val", patch_dir)
    test_ds = PatchDataset("test", patch_dir)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    from src.models.cloud_density import CloudDensityNet

    model = CloudDensityNet(in_channels=3, out_channels=1)
    trainer = DensityTrainer(model)

    patch_dir = PATCHES
    if not patch_dir.exists() or not any(patch_dir.rglob("*.npy")):
        print(f"[WARN] No patches found in {patch_dir}. Generating synthetic data for testing.")
        from src.training.synthetic_data import generate_synthetic_patches
        patch_dir = generate_synthetic_patches()

    train_loader, val_loader, test_loader = create_dataloaders(patch_dir, batch_size=8)
    trainer.fit(train_loader, val_loader, epochs=10)
