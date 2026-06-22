import json
import numpy as np
import rasterio
from pathlib import Path
from dataclasses import dataclass, asdict
from tqdm import tqdm

from src.config import PATCHES, PATCH_SIZE, PATCH_STRIDE, TRAIN_SPLIT, VAL_SPLIT, RANDOM_SEED


@dataclass
class PatchConfig:
    patch_size: int = PATCH_SIZE
    stride: int = PATCH_STRIDE
    min_clear_fraction: float = 0.05
    min_cloud_fraction: float = 0.05
    bands: list = None

    def __post_init__(self):
        if self.bands is None:
            self.bands = [0, 1, 2]


@dataclass
class PatchMetadata:
    scene: str
    patch_id: str
    row: int
    col: int
    x: int
    y: int
    width: int
    height: int
    cloud_fraction: float
    density_class: int
    split: str = ""


class PatchGenerator:
    def __init__(self, config: PatchConfig = None):
        self.config = config or PatchConfig()

    def extract_from_scene(self, image: np.ndarray, mask: np.ndarray, density: np.ndarray,
                           scene_name: str, transform) -> list[tuple[np.ndarray, PatchMetadata]]:
        h, w = image.shape[:2]
        patches = []

        for y in range(0, h - self.config.patch_size + 1, self.config.stride):
            for x in range(0, w - self.config.patch_size + 1, self.config.stride):
                img_patch = image[y:y + self.config.patch_size, x:x + self.config.patch_size, :]
                mask_patch = mask[y:y + self.config.patch_size, x:x + self.config.patch_size]
                density_patch = density[y:y + self.config.patch_size, x:x + self.config.patch_size]

                cloud_frac = float(np.mean(mask_patch))

                if cloud_frac < self.config.min_clear_fraction:
                    continue
                if cloud_frac > (1 - self.config.min_cloud_fraction):
                    continue

                patch_id = f"{scene_name}_y{y}_x{x}"
                meta = PatchMetadata(
                    scene=scene_name,
                    patch_id=patch_id,
                    row=y // self.config.stride,
                    col=x // self.config.stride,
                    x=x,
                    y=y,
                    width=self.config.patch_size,
                    height=self.config.patch_size,
                    cloud_fraction=cloud_frac,
                    density_class=int(np.median(density_patch) * 3),
                )

                patches.append((img_patch, meta))

        return patches

    def split_patches(self, patches: list[PatchMetadata], scene_names: list[str]) -> None:
        rng = np.random.default_rng(RANDOM_SEED)
        unique_scenes = sorted(set(scene_names))
        rng.shuffle(unique_scenes)

        n_train = max(1, int(len(unique_scenes) * TRAIN_SPLIT))
        n_val = max(1, int(len(unique_scenes) * VAL_SPLIT))

        train_scenes = set(unique_scenes[:n_train])
        val_scenes = set(unique_scenes[n_train:n_train + n_val])
        test_scenes = set(unique_scenes[n_train + n_val:])

        for meta in patches:
            if meta.scene in train_scenes:
                meta.split = "train"
            elif meta.scene in val_scenes:
                meta.split = "val"
            else:
                meta.split = "test"

        self._splits = {
            "train": list(train_scenes),
            "val": list(val_scenes),
            "test": list(test_scenes),
        }

    def save_patch(self, patch_data: np.ndarray, metadata: PatchMetadata,
                   out_dir: Path, transform=None, crs=None):
        split_dir = out_dir / metadata.split
        split_dir.mkdir(parents=True, exist_ok=True)

        img_path = split_dir / f"{metadata.patch_id}.npy"
        np.save(str(img_path), patch_data)

        meta_path = split_dir / f"{metadata.patch_id}_meta.json"
        with open(meta_path, "w") as f:
            json.dump(asdict(metadata), f, indent=2)

    def generate_dataset(self, scene_paths: list[Path], mask_paths: list[Path],
                         out_dir: Path = None) -> dict:
        out_dir = Path(out_dir or PATCHES)
        out_dir.mkdir(parents=True, exist_ok=True)

        all_patches_data = []
        all_metas = []
        all_scene_names = []

        for scene_path, mask_path in tqdm(list(zip(scene_paths, mask_paths)), desc="Extracting patches"):
            with rasterio.open(scene_path) as src:
                image = np.moveaxis(src.read(), 0, -1)
                transform = src.transform
                crs = src.crs

            with rasterio.open(mask_path) as src_m:
                mask = src_m.read(1) > 0
                density = src_m.read(2).astype(np.float32) / 255.0

            scene_name = scene_path.stem
            scene_patches = self.extract_from_scene(image, mask, density, scene_name, transform)

            for patch_data, meta in scene_patches:
                all_patches_data.append((patch_data, meta, transform, crs))
                all_metas.append(meta)
                all_scene_names.append(scene_name)

        self.split_patches(all_metas, all_scene_names)

        for patch_data, meta, transform, crs in tqdm(all_patches_data, desc="Saving patches"):
            self.save_patch(patch_data, meta, out_dir, transform, crs)

        summary = {
            "total_patches": len(all_metas),
            "train_patches": sum(1 for m in all_metas if m.split == "train"),
            "val_patches": sum(1 for m in all_metas if m.split == "val"),
            "test_patches": sum(1 for m in all_metas if m.split == "test"),
            "train_scenes": len(self._splits["train"]),
            "val_scenes": len(self._splits["val"]),
            "test_scenes": len(self._splits["test"]),
            "patch_size": self.config.patch_size,
            "stride": self.config.stride,
        }

        with open(out_dir / "dataset_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n[DATASET] Created {summary['total_patches']} patches "
              f"({summary['train_patches']} train, {summary['val_patches']} val, "
              f"{summary['test_patches']} test)")

        return summary


if __name__ == "__main__":
    from src.config import ALIGNED, CLOUD_MASKS

    scenes = sorted(ALIGNED.glob("*.tif")) + sorted(ALIGNED.glob("*.tiff"))
    masks = sorted(CLOUD_MASKS.glob("mask_*.tif"))

    if not scenes or not masks:
        print("[WARN] No scenes or masks found. Run align.py and cloud_mask.py first.")
    else:
        gen = PatchGenerator()
        gen.generate_dataset(scenes, masks)
