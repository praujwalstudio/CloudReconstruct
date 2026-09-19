"""SEN12MS-CR & SEN12MS-CR-TS PyTorch Dataset Implementations (WP0)
================================================================
Grounding Datasets:
1. SEN12MS-CR: https://patricktum.github.io/cloud_removal/sen12mscr/
   122,218 triplets of S1 (VV/VH) + S2 cloudy (13 bands) + S2 clear (13 bands).
2. SEN12MS-CR-TS: https://patricktum.github.io/cloud_removal/sen12mscrts/
   Multi-temporal time series sequences (length t <= 5).

Key Transformations:
1. SAR Normalization: s1_norm = np.clip(s1, -25.0, 0.0) / 12.5 + 1.0 -> [-1.0, 1.0]
2. Optical TOA Reflectance: s2_norm = np.clip(s2 / 10000.0, 0.0, 1.0) -> [0.0, 1.0]
3. LISS-IV Harmonization: Extracts Green (B3/idx 2), Red (B4/idx 3), NIR (B8/idx 7)
4. Strict Non-Overlapping ROI Spatial Partitioning.
"""

import os
import re
from pathlib import Path
from typing import Optional, Callable, Union, List, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
import rasterio

from src.data.band_harmonization import harmonize_s2_to_liss4

try:
    from datasets import load_dataset
    HF_DATASETS_AVAILABLE = True
except ImportError:
    HF_DATASETS_AVAILABLE = False


def normalize_sar(s1_data: np.ndarray) -> np.ndarray:
    """Normalizes SAR backscatter values (dB):
    Clips to [-25.0, 0.0] and scales linearly to [-1.0, 1.0].
    Formula: s1_normalized = np.clip(s1, -25.0, 0.0) / 12.5 + 1.0
    """
    clipped = np.clip(s1_data.astype(np.float32), -25.0, 0.0)
    return clipped / 12.5 + 1.0


def extract_roi_id(path: Union[str, Path]) -> str:
    """Extracts Region of Interest (ROI) identifier from file path or stem
    to enforce strict non-overlapping spatial partitioning."""
    path_obj = Path(path)

    # 1. Check directory path parts
    for part in reversed(path_obj.parts[:-1]):
        if re.search(r"ROIs?\w+", part, re.IGNORECASE):
            return part

    # 2. Check filename stem for ROI pattern
    match = re.search(r"((?:scene_)?ROIs?\w+?)(?:[/_]s[12]|[/_]sigma0|[/_]cloudy|[/_]clear|[/_]patch|[/_]dem|\.tif|$)", path_obj.stem, re.IGNORECASE)
    if match:
        return match.group(1)

    # 3. Fallback to prefix before '_patch'
    stem = path_obj.stem
    if "patch" in stem:
        prefix = stem.split("patch")[0].rstrip("_")
        for suffix in ("_s2_cloudy", "_s2_clear", "_cloudy", "_clear", "_s1", "_s2", "_sigma0", "_dem"):
            if prefix.endswith(suffix):
                prefix = prefix[:-len(suffix)]
        if prefix:
            return prefix

    return path_obj.parent.name or "roi_default"


class SEN12MSCRDataset(Dataset):
    """SEN12MS-CR Mono-Temporal Multi-Modal Dataset.

    Ingests:
    - Sentinel-1 SAR (2 channels: VV, VH in dB) from `sigma0/` or `s1/`
    - Cloudy Sentinel-2 Optical (13 channels) from `cloudy/` or `s2_cloudy/`
    - Clear Target Sentinel-2 Optical (13 channels) from `clear/` or `s2_clear/`
    - Optional Topographic DEM
    """

    def __init__(
        self,
        root_dir: Union[str, Path],
        split: str = "train",
        transform: Optional[Callable] = None,
        return_full_s2: bool = False,
        return_dict: bool = False,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42,
    ):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.split = split.lower()
        self.transform = transform
        self.return_full_s2 = return_full_s2
        self.return_dict = return_dict
        self.seed = seed

        if self.split not in ("train", "val", "test", "all"):
            raise ValueError(f"Invalid split: {split}. Expected 'train', 'val', 'test', or 'all'.")

        self.samples = self._discover_and_split_samples(train_ratio, val_ratio)

    def _discover_and_split_samples(self, train_ratio: float, val_ratio: float) -> list[dict]:
        """Discovers triplet files across `sigma0/`, `cloudy/`, `clear/` or ROI subfolders."""
        if not self.root_dir.exists():
            return []

        cloudy_candidates = set(self.root_dir.rglob("*s2_cloudy*.tif*")) | set(self.root_dir.rglob("*cloudy*.tif*"))
        if not cloudy_candidates:
            cloudy_dir = self.root_dir / "cloudy"
            if cloudy_dir.exists():
                cloudy_candidates = set(cloudy_dir.rglob("*.tif*"))

        if not cloudy_candidates:
            cloudy_candidates = {
                f for f in self.root_dir.rglob("*.tif*")
                if not any(k in f.name.lower() for k in ("s1", "sigma0", "clear", "dem", "mask"))
            }

        s2_cloudy_files = sorted(list(cloudy_candidates))

        samples_by_roi: Dict[str, list] = {}
        for cloudy_path in s2_cloudy_files:
            roi_id = extract_roi_id(cloudy_path)
            dir_path = cloudy_path.parent
            base_name = cloudy_path.name

            # 1. Resolve s2_clear
            clear_name = base_name.replace("s2_cloudy", "s2_clear").replace("cloudy", "clear")
            clear_path = dir_path / clear_name
            if not clear_path.exists():
                sibling_clear = dir_path.parent / "clear" / clear_name
                if sibling_clear.exists():
                    clear_path = sibling_clear
                else:
                    candidates = list(self.root_dir.rglob(f"*{clear_name}*"))
                    clear_path = candidates[0] if candidates else cloudy_path

            # 2. Resolve s1 (SAR / sigma0)
            # Order matters: try specific replacements first
            s1_name_candidates = [
                base_name.replace("cloudy", "sigma0"),  # Most specific: cloudy -> sigma0
                base_name.replace("s2_cloudy", "s1"),
                base_name.replace("cloudy", "s1"),
                base_name.replace("s2", "s1"),
            ]
            s1_path = None
            found_cand = None
            for s1_cand in s1_name_candidates:
                cand_path = dir_path / s1_cand
                if cand_path.exists():
                    s1_path = cand_path
                    found_cand = s1_cand
                    break
                sibling_cand = dir_path.parent / "sigma0" / s1_cand
                if sibling_cand.exists():
                    s1_path = sibling_cand
                    found_cand = s1_cand
                    break
            if s1_path is None and found_cand is not None:
                matches = list(self.root_dir.rglob(f"*{found_cand}*"))
                s1_path = matches[0] if matches else None

            # 3. Resolve DEM
            dem_name_candidates = [
                base_name.replace("cloudy", "dem"),  # Most specific first
                base_name.replace("s2_cloudy", "dem"),
                base_name.replace("s2", "dem"),
            ]
            dem_path = None
            found_dem_cand = None
            for dem_cand in dem_name_candidates:
                cand_path = dir_path / dem_cand
                if cand_path.exists():
                    dem_path = cand_path
                    found_dem_cand = dem_cand
                    break
                sibling_cand = dir_path.parent / "dem" / dem_cand
                if sibling_cand.exists():
                    dem_path = sibling_cand
                    found_dem_cand = dem_cand
                    break
            if dem_path is None and found_dem_cand is not None:
                matches = list(self.root_dir.rglob(f"*{found_dem_cand}*"))
                dem_path = matches[0] if matches else None

            sample_entry = {
                "roi_id": roi_id,
                "s2_cloudy_path": cloudy_path,
                "s2_clear_path": clear_path,
                "s1_path": s1_path,
                "dem_path": dem_path,
            }

            if roi_id not in samples_by_roi:
                samples_by_roi[roi_id] = []
            samples_by_roi[roi_id].append(sample_entry)

        # Enforce strict ROI non-overlapping split
        unique_rois = sorted(list(samples_by_roi.keys()))
        rng = np.random.default_rng(self.seed)
        shuffled_rois = unique_rois.copy()
        rng.shuffle(shuffled_rois)

        n_rois = len(shuffled_rois)
        n_train = int(n_rois * train_ratio)
        n_val = int(n_rois * val_ratio)
        if n_rois >= 3 and n_train == n_rois:
            n_train = n_rois - 2
            n_val = 1

        train_rois = set(shuffled_rois[:n_train])
        val_rois = set(shuffled_rois[n_train:n_train + n_val])
        test_rois = set(shuffled_rois[n_train + n_val:])

        if self.split == "train":
            active_rois = train_rois
        elif self.split == "val":
            active_rois = val_rois
        elif self.split == "test":
            active_rois = test_rois
        else:
            active_rois = set(unique_rois)

        split_samples = []
        for roi in unique_rois:
            if roi in active_rois:
                split_samples.extend(samples_by_roi[roi])

        return split_samples

    def __len__(self) -> int:
        return len(self.samples)

    def _read_geotiff(self, path: Path) -> Tuple[np.ndarray, dict]:
        """Reads a GeoTIFF defensively using rasterio."""
        with rasterio.open(path) as src:
            data = src.read().astype(np.float32)  # (C, H, W)
            meta = {
                "crs": str(src.crs),
                "transform": [float(x) for x in list(src.transform)[:6]],
                "bounds": [float(b) for b in src.bounds],
                "count": int(src.count),
                "nodata": float(src.nodata) if src.nodata is not None else 0.0,
            }
            return data, meta

    def __getitem__(self, idx: int) -> Union[tuple, dict]:
        sample = self.samples[idx]

        # 1. Load S2 cloudy (13 channels)
        s2_cloudy_raw, meta = self._read_geotiff(sample["s2_cloudy_path"])

        # 2. Load S2 clear (13 channels)
        if sample["s2_clear_path"] and sample["s2_clear_path"].exists():
            s2_clear_raw, _ = self._read_geotiff(sample["s2_clear_path"])
        else:
            s2_clear_raw = s2_cloudy_raw.copy()

        # 3. Load S1 SAR (VV, VH in dB)
        if sample["s1_path"] and sample["s1_path"].exists():
            s1_raw, _ = self._read_geotiff(sample["s1_path"])
            if s1_raw.shape[0] > 2:
                s1_raw = s1_raw[:2, :, :]
        else:
            _, h, w = s2_cloudy_raw.shape
            s1_raw = np.zeros((2, h, w), dtype=np.float32)

        # 4. Harmonize S2 bands to LISS-IV (Green: idx 2, Red: idx 3, NIR: idx 7)
        if self.return_full_s2:
            s2_cloudy = np.clip(s2_cloudy_raw / 10000.0, 0.0, 1.0)
            s2_clear = np.clip(s2_clear_raw / 10000.0, 0.0, 1.0)
        else:
            s2_cloudy = harmonize_s2_to_liss4(s2_cloudy_raw, scale_toa=True, clip_range=(0.0, 1.0))
            s2_clear = harmonize_s2_to_liss4(s2_clear_raw, scale_toa=True, clip_range=(0.0, 1.0))

        # 5. SAR Normalization: [-25, 0] -> [-1, 1]
        s1_norm = normalize_sar(s1_raw)

        s2_cloudy_t = torch.from_numpy(s2_cloudy).float()
        s2_clear_t = torch.from_numpy(s2_clear).float()
        s1_t = torch.from_numpy(s1_norm).float()

        dem_t = None
        if sample["dem_path"] and sample["dem_path"].exists():
            dem_raw, _ = self._read_geotiff(sample["dem_path"])
            dem_t = torch.from_numpy(dem_raw).float()

        if self.transform is not None:
            s2_cloudy_t, s2_clear_t = self.transform(s2_cloudy_t, s2_clear_t)

        if self.return_dict:
            res = {
                "s2_cloudy": s2_cloudy_t,
                "s2_clear": s2_clear_t,
                "s1": s1_t,
                "roi_id": sample["roi_id"],
                "path": str(sample["s2_cloudy_path"]),
                "meta": meta,
            }
            if dem_t is not None:
                res["dem"] = dem_t
            return res

        return s2_cloudy_t, s2_clear_t, s1_t


class SEN12MSCRTSDataset(Dataset):
    """SEN12MS-CR-TS Multi-Temporal Time-Series Dataset.
    Loads sequences of cloudy and clear observations (sequence length t <= 5).
    """

    def __init__(
        self,
        root_dir: Union[str, Path],
        seq_length: int = 5,
        split: str = "train",
        transform: Optional[Callable] = None,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42,
    ):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.seq_length = seq_length
        self.split = split.lower()
        self.transform = transform
        self.seed = seed
        self.sequences = self._discover_sequences(train_ratio, val_ratio)

    def _discover_sequences(self, train_ratio: float, val_ratio: float) -> list[list[dict]]:
        base_ds = SEN12MSCRDataset(
            self.root_dir,
            split=self.split,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=self.seed,
        )

        grouped = {}
        for sample in base_ds.samples:
            roi = sample["roi_id"]
            if roi not in grouped:
                grouped[roi] = []
            grouped[roi].append(sample)

        sequences = []
        for roi, samples in grouped.items():
            for i in range(0, len(samples), self.seq_length):
                chunk = samples[i:i + self.seq_length]
                if len(chunk) > 0:
                    sequences.append(chunk)

        return sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict:
        seq_samples = self.sequences[idx]
        cloudy_list, clear_list, s1_list = [], [], []

        for sample in seq_samples:
            with rasterio.open(sample["s2_cloudy_path"]) as src:
                raw_c = src.read().astype(np.float32)
            c_liss4 = harmonize_s2_to_liss4(raw_c, scale_toa=True)
            cloudy_list.append(torch.from_numpy(c_liss4).float())

            if sample["s2_clear_path"] and sample["s2_clear_path"].exists():
                with rasterio.open(sample["s2_clear_path"]) as src:
                    raw_cl = src.read().astype(np.float32)
                cl_liss4 = harmonize_s2_to_liss4(raw_cl, scale_toa=True)
            else:
                cl_liss4 = c_liss4
            clear_list.append(torch.from_numpy(cl_liss4).float())

            if sample["s1_path"] and sample["s1_path"].exists():
                with rasterio.open(sample["s1_path"]) as src:
                    s1_raw = src.read()[:2, :, :].astype(np.float32)
                s1_norm = normalize_sar(s1_raw)
            else:
                s1_norm = np.zeros((2, raw_c.shape[1], raw_c.shape[2]), dtype=np.float32)
            s1_list.append(torch.from_numpy(s1_norm).float())

        return {
            "cloudy_seq": torch.stack(cloudy_list, dim=0),  # (T, 3, H, W)
            "clear_seq": torch.stack(clear_list, dim=0),    # (T, 3, H, W)
            "s1_seq": torch.stack(s1_list, dim=0),          # (T, 2, H, W)
            "seq_len": len(cloudy_list),
            "roi_id": seq_samples[0]["roi_id"],
        }


class SEN12MSCRStreamingDataset(Dataset):
    """SEN12MS-CR Streaming Dataset using HF datasets library.
    
    Fetches samples on-demand from HuggingFace Hub. No local storage required.
    Uses the same preprocessing as SEN12MSCRDataset.
    """
    
    def __init__(
        self,
        split: str = "train",
        transform: Optional[Callable] = None,
        return_full_s2: bool = False,
        return_dict: bool = False,
        cache_dir: Optional[str] = None,
        shuffle_buffer: int = 0,
    ):
        super().__init__()
        if not HF_DATASETS_AVAILABLE:
            raise ImportError("datasets library required: pip install datasets")
        
        self.split = split.lower()
        self.transform = transform
        self.return_full_s2 = return_full_s2
        self.return_dict = return_dict
        self.cache_dir = cache_dir
        self.shuffle_buffer = shuffle_buffer
        
        if self.split not in ("train", "validation", "test"):
            raise ValueError(f"Invalid split: {split}. Expected 'train', 'validation', or 'test'.")
        
        # Load streaming dataset
        self._ds = load_dataset(
            "Hermanni/sen12mscr",
            split=self.split,
            streaming=True,
            cache_dir=cache_dir,
        )
        # Get length by iterating once (only for train/val/test splits we know the sizes)
        # For now, we'll iterate to count - in practice use known sizes
        self._length = None
    
    def __len__(self) -> int:
        # Known sizes from dataset card
        sizes = {"train": 107036, "validation": 7184, "test": 7998}
        return sizes.get(self.split, 0)
    
    def __getitem__(self, idx: int) -> Union[tuple, dict]:
        # Streaming dataset doesn't support random access by index directly
        # We need to iterate to the idx. For training with shuffling, we'll use
        # a different approach - wrap in IterableDataset or use DataLoader with
        # a custom sampler. For now, implement sequential access.
        # Note: This is not efficient for random access. Better to use
        # an IterableDataset wrapper for training.
        if self._length is None:
            # This shouldn't happen in normal use
            raise RuntimeError("Streaming dataset requires IterableDataset wrapper for training")
        
        # Fallback for validation/test: iterate to idx
        it = iter(self._ds)
        for i, sample in enumerate(it):
            if i == idx:
                return self._process_sample(sample)
        raise IndexError(f"Index {idx} out of range")
    
    def _process_sample(self, sample: dict) -> Union[tuple, dict]:
        """Process a raw HF sample into model inputs."""
        # Decode SAR (HWC -> CHW)
        sar = np.frombuffer(sample["sar"], dtype=np.float32).reshape(sample["sar_shape"])
        sar = sar.transpose(2, 0, 1)  # (2, 256, 256)
        
        # Decode optical (HWC)
        cloudy = np.frombuffer(sample["cloudy"], dtype=np.int16).reshape(sample["opt_shape"])
        target = np.frombuffer(sample["target"], dtype=np.int16).reshape(sample["opt_shape"])
        
        # Select LISS-IV bands (G=2, R=3, NIR=7) and transpose to CHW
        cloudy_liss4 = cloudy[:, :, [2, 3, 7]].transpose(2, 0, 1).astype(np.float32)
        target_liss4 = target[:, :, [2, 3, 7]].transpose(2, 0, 1).astype(np.float32)
        
        # Normalize optical: DN / 10000 -> [0, 1]
        cloudy_liss4 = np.clip(cloudy_liss4 / 10000.0, 0.0, 1.0)
        target_liss4 = np.clip(target_liss4 / 10000.0, 0.0, 1.0)
        
        # Normalize SAR: clip [-25, 0] -> [-1, 1]
        sar = np.clip(sar, -25.0, 0.0)
        sar = sar / 12.5 + 1.0
        
        s2_cloudy_t = torch.from_numpy(cloudy_liss4).float()
        s2_clear_t = torch.from_numpy(target_liss4).float()
        s1_t = torch.from_numpy(sar).float()
        
        if self.transform is not None:
            s2_cloudy_t, s2_clear_t = self.transform(s2_cloudy_t, s2_clear_t)
        
        if self.return_dict:
            return {
                "s2_cloudy": s2_cloudy_t,
                "s2_clear": s2_clear_t,
                "s1": s1_t,
                "roi_id": f"{sample['season']}_scene{sample['scene']}",
                "patch_id": sample["patch"],
            }
        
        return s2_cloudy_t, s2_clear_t, s1_t
    
    def as_iterable(self):
        """Returns an iterable for use with DataLoader."""
        ds = self._ds
        if self.shuffle_buffer and self.shuffle_buffer > 0:
            ds = ds.shuffle(buffer_size=self.shuffle_buffer, seed=42)
        for sample in ds:
            yield self._process_sample(sample)


class SEN12MSCRStreamingIterable(torch.utils.data.IterableDataset):
    """IterableDataset wrapper for SEN12MSCR streaming with proper shuffling."""
    
    def __init__(
        self,
        split: str = "train",
        transform: Optional[Callable] = None,
        return_full_s2: bool = False,
        return_dict: bool = False,
        cache_dir: Optional[str] = None,
        shuffle_buffer: int = 50,
    ):
        super().__init__()
        if not HF_DATASETS_AVAILABLE:
            raise ImportError("datasets library required: pip install datasets")
        
        self.split = split.lower()
        self.transform = transform
        self.return_full_s2 = return_full_s2
        self.return_dict = return_dict
        self.cache_dir = cache_dir
        self.shuffle_buffer = shuffle_buffer
        
        self._ds = load_dataset(
            "Hermanni/sen12mscr",
            split=self.split,
            streaming=True,
            cache_dir=cache_dir,
        )
    
    def _process_sample(self, sample: dict):
        sar = np.frombuffer(sample["sar"], dtype=np.float32).reshape(sample["sar_shape"])
        sar = sar.transpose(2, 0, 1)
        
        cloudy = np.frombuffer(sample["cloudy"], dtype=np.int16).reshape(sample["opt_shape"])
        target = np.frombuffer(sample["target"], dtype=np.int16).reshape(sample["opt_shape"])
        
        cloudy_liss4 = cloudy[:, :, [2, 3, 7]].transpose(2, 0, 1).astype(np.float32)
        target_liss4 = target[:, :, [2, 3, 7]].transpose(2, 0, 1).astype(np.float32)
        
        cloudy_liss4 = np.clip(cloudy_liss4 / 10000.0, 0.0, 1.0)
        target_liss4 = np.clip(target_liss4 / 10000.0, 0.0, 1.0)
        
        sar = np.clip(sar, -25.0, 0.0)
        sar = sar / 12.5 + 1.0
        
        s2_cloudy_t = torch.from_numpy(cloudy_liss4).float()
        s2_clear_t = torch.from_numpy(target_liss4).float()
        s1_t = torch.from_numpy(sar).float()
        
        if self.transform is not None:
            s2_cloudy_t, s2_clear_t = self.transform(s2_cloudy_t, s2_clear_t)
        
        if self.return_dict:
            return {
                "s2_cloudy": s2_cloudy_t,
                "s2_clear": s2_clear_t,
                "s1": s1_t,
                "roi_id": f"{sample['season']}_scene{sample['scene']}",
                "patch_id": sample["patch"],
            }
        
        return s2_cloudy_t, s2_clear_t, s1_t
    
    def __iter__(self):
        # Shuffle at the dataset level (HF handles this efficiently)
        ds = self._ds.shuffle(buffer_size=self.shuffle_buffer, seed=42)
        for sample in ds:
            yield self._process_sample(sample)
    
    def __len__(self):
        sizes = {"train": 107036, "validation": 7184, "test": 7998}
        return sizes.get(self.split, 0)


class CompactSEN12MSDataset(Dataset):
    """High-performance dataset loader for local compact .npz satellite scenes.

    Ingests pre-converted multi-modal scene archives:
    - 3-band LISS-IV optical cloudy (Green B3, Red B4, NIR B8)
    - 3-band LISS-IV optical clear target (Green B3, Red B4, NIR B8)
    - 2-band Sentinel-1 SAR (VV, VH in dB)
    """

    def __init__(
        self,
        root_dir: Union[str, Path] = "data/raw/sen12ms_cr/compact",
        split: str = "train",
        transform: Optional[Callable] = None,
        return_dict: bool = False,
        preload_memory: bool = False,
    ):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.split = "val" if split.lower() in ("val", "validation") else split.lower()
        self.transform = transform
        self.return_dict = return_dict
        self.preload_memory = preload_memory

        # Find manifest or scan directories
        self.index: List[Tuple[Path, int, str]] = []  # (npz_path, patch_idx_in_file, scene_name)
        self._open_archives: Dict[str, dict] = {}
        self._build_index()

    def _build_index(self):
        manifest_path = self.root_dir / "manifest.json"
        if not manifest_path.exists() and (self.root_dir / "compact" / "manifest.json").exists():
            manifest_path = self.root_dir / "compact" / "manifest.json"

        if manifest_path.exists():
            try:
                import json
                with open(manifest_path, "r") as f:
                    manifest = json.load(f)
                split_entries = manifest.get(self.split, [])
                base_dir = self.root_dir.parent if self.root_dir.name == "compact" else self.root_dir
                for entry in split_entries:
                    rel_p = entry["npz_path"]
                    npz_file = base_dir / rel_p
                    if not npz_file.exists():
                        npz_file = self.root_dir / rel_p
                    if npz_file.exists():
                        n_p = entry.get("n_patches", 0)
                        scene_id = f"{entry.get('season', 'season')}_scene{entry.get('scene', 0)}"
                        for i in range(n_p):
                            self.index.append((npz_file, i, scene_id))
            except Exception as e:
                pass

        if not self.index:
            # Fallback: discover all .npz in target split directory
            split_dirs = [self.root_dir / self.split, self.root_dir / "compact" / self.split]
            for s_dir in split_dirs:
                if s_dir.exists():
                    for npz_file in sorted(s_dir.rglob("*.npz")):
                        try:
                            archive = np.load(npz_file, allow_pickle=True)
                            n_p = len(archive["cloudy"])
                            scene_id = f"{npz_file.parent.name}_{npz_file.stem}"
                            for i in range(n_p):
                                self.index.append((npz_file, i, scene_id))
                        except Exception:
                            continue

    def __len__(self) -> int:
        return len(self.index)

    def _get_archive_data(self, npz_path: Path):
        key = str(npz_path)
        if key not in self._open_archives:
            archive = np.load(npz_path, allow_pickle=True)
            self._open_archives[key] = {
                "cloudy": archive["cloudy"],
                "target": archive["target"],
                "sar": archive["sar"],
            }
        return self._open_archives[key]

    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], dict]:
        npz_file, patch_idx, scene_id = self.index[idx]
        data = self._get_archive_data(npz_file)

        raw_cloudy = data["cloudy"][patch_idx].astype(np.float32)
        raw_target = data["target"][patch_idx].astype(np.float32)
        raw_sar = data["sar"][patch_idx].astype(np.float32)

        # Scale optical DN -> TOA Reflectance [0.0, 1.0]
        cloudy = np.clip(raw_cloudy / 10000.0, 0.0, 1.0)
        target = np.clip(raw_target / 10000.0, 0.0, 1.0)

        # Normalize SAR backscatter [-25, 0] dB -> [-1.0, 1.0]
        sar = np.clip(raw_sar, -25.0, 0.0) / 12.5 + 1.0

        cloudy_t = torch.from_numpy(cloudy).float()
        target_t = torch.from_numpy(target).float()
        sar_t = torch.from_numpy(sar).float()

        if self.transform is not None:
            cloudy_t, target_t = self.transform(cloudy_t, target_t)

        if self.return_dict:
            return {
                "s2_cloudy": cloudy_t,
                "s2_clear": target_t,
                "s1": sar_t,
                "roi_id": scene_id,
                "patch_id": patch_idx,
            }

        return cloudy_t, target_t, sar_t
