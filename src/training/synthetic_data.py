import json
import numpy as np
from pathlib import Path
from src.config import PATCHES, RANDOM_SEED


def generate_synthetic_patches(out_dir: Path = None, n_scenes: int = 5,
                                patches_per_scene: int = 20,
                                with_sar: bool = False,
                                with_temporal_ref: bool = False) -> Path:
    out_dir = Path(out_dir or PATCHES)
    rng = np.random.default_rng(RANDOM_SEED)

    splits = {"train": [], "val": [], "test": []}
    scene_names = [f"synth_scene_{i}" for i in range(n_scenes)]

    for i, scene_name in enumerate(scene_names):
        train_limit = int(n_scenes * 0.8)
        val_limit = int(n_scenes * 0.9)
        if val_limit == train_limit and n_scenes >= 3:
            val_limit = train_limit + 1
        split = "train" if i < train_limit else ("val" if i < val_limit else "test")
        split_dir = out_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

        for j in range(patches_per_scene):
            patch_id = f"{scene_name}_y{j}_x0"
            h, w = 256, 256

            ground = rng.integers(100, 600, (h, w, 3), dtype=np.uint16)
            cloud_frac = rng.uniform(0.05, 0.95)
            cloud_mask = rng.random((h, w)) < cloud_frac
            cloud_density_val = rng.uniform(0.5, 1.0)
            image = ground.copy()
            image[cloud_mask] = np.clip(
                image[cloud_mask].astype(np.float32) * (1.0 + cloud_density_val), 0, 1023
            ).astype(np.uint16)

            np.save(str(split_dir / f"{patch_id}.npy"), image)

            if with_sar:
                sar = rng.integers(0, 500, (h, w, 2), dtype=np.uint16)
                np.save(str(split_dir / f"{patch_id}_sar.npy"), sar)

            if with_temporal_ref:
                ref_cloud_frac = rng.uniform(0.0, 0.3)
                ref_cloud_mask = rng.random((h, w)) < ref_cloud_frac
                ref_image = ground.copy()
                ref_image[ref_cloud_mask] = np.clip(
                    ref_image[ref_cloud_mask].astype(np.float32) * (1.0 + rng.uniform(0.3, 0.7)), 0, 1023
                ).astype(np.uint16)
                np.save(str(split_dir / f"{patch_id}_ref.npy"), ref_image)

            meta = {
                "scene": scene_name,
                "patch_id": patch_id,
                "cloud_fraction": float(cloud_frac),
                "split": split,
                "has_sar": with_sar,
                "has_temporal_ref": with_temporal_ref,
            }
            with open(split_dir / f"{patch_id}_meta.json", "w") as f:
                json.dump(meta, f, indent=2)

            splits[split].append(patch_id)

    summary = {
        "total": sum(len(v) for v in splits.values()),
        "train": len(splits["train"]),
        "val": len(splits["val"]),
        "test": len(splits["test"]),
        "has_sar": with_sar,
        "has_temporal_ref": with_temporal_ref,
    }
    print(f"[SYNTH] Generated {summary['total']} synthetic patches "
          f"({summary['train']}/{summary['val']}/{summary['test']} split)"
          + (" + SAR" if with_sar else "")
          + (" + temporal refs" if with_temporal_ref else ""))
    return out_dir
