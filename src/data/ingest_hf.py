"""SEN12MS-CR HuggingFace Mirror Ingestion (WP0 Revised)
=========================================================
Downloads parquet from Hermanni/sen12mscr, converts to LISS-IV bands + SAR,
writes compact per-scene .npz, deletes parquet. Resume-safe.
"""

import argparse
import json
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests
from tqdm import tqdm

try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None

HF_REPO = "Hermanni/sen12mscr"
HF_BASE_URL = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main"

# Scene manifests from dataset card (train/val/test splits by scene)
SCENE_MANIFEST = {
    "train": {
        "spring": [1, 6, 8, 9, 15, 21, 26, 39, 40, 45, 58, 63, 66, 75, 77, 97, 100, 101, 109, 110, 113, 115, 117, 119, 120, 121, 124, 126, 128, 132, 134, 141, 142, 145, 147],
        "summer": [4, 7, 11, 15, 25, 27, 31, 36, 40, 42, 43, 47, 55, 56, 72, 76, 86, 87, 89, 90, 93, 95, 100, 101, 102, 113, 114, 115, 120, 121, 123, 124, 125, 126, 132, 133, 135, 137, 139, 140, 143, 146, 147],
        "fall": [1, 3, 4, 6, 11, 14, 19, 22, 26, 27, 28, 30, 31, 33, 35, 37, 39, 40, 41, 42, 57, 64, 71, 77, 81, 82, 83, 85, 88, 91, 93, 100, 104, 105, 107, 109, 110, 112, 114, 116, 119, 120, 122, 125, 128, 131, 133, 134, 135, 136, 141, 142, 144, 147, 148, 149],
        "winter": [8, 21, 25, 42, 47, 49, 55, 59, 61, 62, 64, 68, 75, 81, 94, 102, 104, 112, 116, 135, 146],
    },
    "val": {
        "spring": [17],
        "summer": [17, 19, 80, 127],
        "fall": [65],
        "winter": [22, 84, 107, 130],
    },
    "test": {
        "spring": [31, 44, 106, 123, 140],
        "summer": [73, 119],
        "fall": [139],
        "winter": [63, 108],
    },
}

# LISS-IV band indices in S2 13-band (0-indexed): B3(Green)=2, B4(Red)=3, B8(NIR)=7
LISS4_INDICES = [2, 3, 7]

PROGRESS_FILE = "ingest_progress.json"
MANIFEST_FILE = "manifest.json"


def parse_args():
    parser = argparse.ArgumentParser(description="SEN12MS-CR HF Mirror Ingestion")
    parser.add_argument("--raw-dir", type=str, default="data/raw/sen12ms_cr_hf",
                        help="Directory for downloaded parquet files")
    parser.add_argument("--out-dir", type=str, default="data/raw/sen12ms_cr",
                        help="Output directory for compact npz store")
    parser.add_argument("--seasons", nargs="+", default=None,
                        choices=["spring", "summer", "fall", "winter"],
                        help="Seasons to process (default: all)")
    parser.add_argument("--splits", nargs="+", default=None,
                        choices=["train", "val", "test"],
                        help="Splits to process (default: all)")
    parser.add_argument("--workers", type=int, default=6,
                        help="Parallel download workers")
    parser.add_argument("--limit-scenes", type=int, default=None,
                        help="Limit scenes per season/split (for testing)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print planned actions without downloading")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from progress file")
    parser.add_argument("--inspect-only", action="store_true",
                        help="Download first scene only, inspect ranges, exit")
    return parser.parse_args()


def load_progress(raw_dir: Path) -> Dict:
    prog_path = raw_dir / PROGRESS_FILE
    if prog_path.exists():
        with open(prog_path) as f:
            return json.load(f)
    return {"completed": [], "failed": []}


def save_progress(raw_dir: Path, progress: Dict):
    prog_path = raw_dir / PROGRESS_FILE
    with open(prog_path, "w") as f:
        json.dump(progress, f, indent=2)


def load_manifest(out_dir: Path) -> Dict:
    man_path = out_dir / "compact" / MANIFEST_FILE
    if man_path.exists():
        with open(man_path) as f:
            return json.load(f)
    return {"train": [], "val": [], "test": []}


def save_manifest(out_dir: Path, manifest: Dict):
    man_path = out_dir / "compact" / MANIFEST_FILE
    man_path.parent.mkdir(parents=True, exist_ok=True)
    with open(man_path, "w") as f:
        json.dump(manifest, f, indent=2)


def download_parquet(url: str, dest: Path, chunk_size: int = 1024 * 1024) -> bool:
    """Download with HTTP Range resume support."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest.with_suffix(dest.suffix + ".part")

    initial_size = temp_path.stat().st_size if temp_path.exists() else 0
    headers = {"Range": f"bytes={initial_size}-"} if initial_size > 0 else {}

    try:
        resp = requests.get(url, headers=headers, stream=True, timeout=120, allow_redirects=True)
        if resp.status_code == 416:
            temp_path.rename(dest)
            return True
        if resp.status_code not in (200, 206):
            print(f"  [WARN] HTTP {resp.status_code} for {dest.name}")
            return False

        total_size = int(resp.headers.get("content-length", 0)) + initial_size
        mode = "ab" if initial_size > 0 else "wb"

        with open(temp_path, mode) as f, tqdm(
            total=total_size, initial=initial_size,
            unit="B", unit_scale=True, unit_divisor=1024,
            desc=dest.name, leave=False
        ) as pbar:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

        temp_path.rename(dest)
        return True
    except Exception as e:
        print(f"  [ERROR] Download failed for {dest.name}: {e}")
        return False


def inspect_parquet(parquet_path: Path) -> Tuple[dict, int]:
    """Read first row to inspect value ranges."""
    if pq is None:
        raise ImportError("pyarrow required: pip install pyarrow")
    table = pq.read_table(parquet_path)
    sar_bytes = table["sar"][0].as_py()
    cloudy_bytes = table["cloudy"][0].as_py()
    target_bytes = table["target"][0].as_py()
    sar_shape = table["sar_shape"][0].as_py()
    opt_shape = table["opt_shape"][0].as_py()

    sar = np.frombuffer(sar_bytes, dtype=np.float32).reshape(sar_shape)
    cloudy = np.frombuffer(cloudy_bytes, dtype=np.int16).reshape(opt_shape)
    target = np.frombuffer(target_bytes, dtype=np.int16).reshape(opt_shape)
    stats = {
        "sar_shape": sar.shape,
        "sar_min": float(sar.min()),
        "sar_max": float(sar.max()),
        "sar_mean": float(sar.mean()),
        "opt_shape": cloudy.shape,
        "cloudy_min": int(cloudy.min()),
        "cloudy_max": int(cloudy.max()),
        "target_min": int(target.min()),
        "target_max": int(target.max()),
    }
    return stats, len(table)


def convert_parquet_to_npz(
    parquet_path: Path,
    out_dir: Path,
    split: str,
    season: str,
    scene_num: int,
    sar_dtype: np.dtype,
) -> Optional[Dict]:
    """Convert a parquet file to compact npz."""
    if pq is None:
        raise ImportError("pyarrow required: pip install pyarrow")

    table = pq.read_table(parquet_path)
    n_rows = len(table)
    if n_rows == 0:
        print(f"  [WARN] Empty parquet: {parquet_path.name}")
        return None

    # Pre-allocate arrays
    cloudy_arr = np.empty((n_rows, 3, 256, 256), dtype=np.int16)
    target_arr = np.empty((n_rows, 3, 256, 256), dtype=np.int16)
    sar_arr = np.empty((n_rows, 2, 256, 256), dtype=sar_dtype)
    patch_ids = []

    pydict = table.to_pydict()
    sar_list = pydict["sar"]
    cloudy_list = pydict["cloudy"]
    target_list = pydict["target"]
    sar_shapes = pydict["sar_shape"]
    opt_shapes = pydict["opt_shape"]
    patch_list = pydict["patch"]

    for i in range(n_rows):
        sar = np.frombuffer(sar_list[i], dtype=np.float32).reshape(sar_shapes[i])
        cloudy = np.frombuffer(cloudy_list[i], dtype=np.int16).reshape(opt_shapes[i])
        target = np.frombuffer(target_list[i], dtype=np.int16).reshape(opt_shapes[i])
        patch_id = patch_list[i]

        # Select LISS-IV bands (G=2, R=3, NIR=7) from HWC -> CHW
        cloudy_liss4 = cloudy[:, :, LISS4_INDICES].transpose(2, 0, 1)  # (3, 256, 256)
        target_liss4 = target[:, :, LISS4_INDICES].transpose(2, 0, 1)

        # SAR: store as inspected dtype (float32 or uint8)
        if sar_dtype == np.uint8:
            sar_final = np.clip(sar, 0, 255).astype(np.uint8)
        else:
            sar_final = sar.astype(np.float32)

        if sar_final.ndim == 3 and sar_final.shape[-1] == 2:
            sar_final = sar_final.transpose(2, 0, 1)  # (2, 256, 256)

        cloudy_arr[i] = cloudy_liss4
        target_arr[i] = target_liss4
        sar_arr[i] = sar_final
        patch_ids.append(patch_id)

    # Write per-scene npz (uncompressed for fast loading)
    npz_path = out_dir / "compact" / split / season / f"scene_{scene_num}.npz"
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        npz_path,
        cloudy=cloudy_arr,
        target=target_arr,
        sar=sar_arr,
        patch_ids=np.array(patch_ids, dtype=object),
    )

    return {
        "split": split,
        "season": season,
        "scene": scene_num,
        "npz_path": str(npz_path.relative_to(out_dir)),
        "n_patches": n_rows,
    }


def process_scene(
    split: str,
    season: str,
    scene_num: int,
    raw_dir: Path,
    out_dir: Path,
    sar_dtype: np.dtype,
    progress: Dict,
    manifest: Dict,
    dry_run: bool = False,
) -> bool:
    scene_key = f"{split}/{season}/scene_{scene_num}"
    if scene_key in progress["completed"]:
        print(f"  [SKIP] {scene_key} already completed")
        return True

    parquet_name = f"scene_{scene_num}.parquet"
    parquet_url = f"{HF_BASE_URL}/{season}/{parquet_name}"
    parquet_path = raw_dir / season / parquet_name

    if dry_run:
        print(f"  [DRY-RUN] Would download {parquet_url} -> {parquet_path}")
        return True

    print(f"  [DOWNLOAD] {scene_key}")
    if not download_parquet(parquet_url, parquet_path):
        progress["failed"].append(scene_key)
        save_progress(raw_dir, progress)
        return False

    print(f"  [CONVERT] {scene_key}")
    try:
        entry = convert_parquet_to_npz(parquet_path, out_dir, split, season, scene_num, sar_dtype)
        if entry:
            manifest[split].append(entry)
            save_manifest(out_dir, manifest)
    except Exception as e:
        print(f"  [ERROR] Conversion failed for {scene_key}: {e}")
        progress["failed"].append(scene_key)
        save_progress(raw_dir, progress)
        return False

    # Delete parquet to save space
    try:
        parquet_path.unlink()
    except OSError:
        pass

    progress["completed"].append(scene_key)
    save_progress(raw_dir, progress)
    print(f"  [DONE] {scene_key} ({entry['n_patches']} patches)")
    return True


def main():
    args = parse_args()

    if pq is None:
        print("[ERROR] pyarrow not installed. Run: pip install pyarrow")
        sys.exit(1)

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)

    seasons = args.seasons or ["spring", "summer", "fall", "winter"]
    splits = args.splits or ["train", "val", "test"]

    raw_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    progress = load_progress(raw_dir) if args.resume else {"completed": [], "failed": []}
    manifest = load_manifest(out_dir)

    # Build work list
    work = []
    for split in splits:
        for season in seasons:
            scenes = SCENE_MANIFEST[split].get(season, [])
            if args.limit_scenes:
                scenes = scenes[:args.limit_scenes]
            for scene_num in scenes:
                work.append((split, season, scene_num))

    if args.dry_run:
        print(f"Planned: {len(work)} scenes across splits={splits}, seasons={seasons}")
        for split, season, scene in work:
            print(f"  {split}/{season}/scene_{scene}")
        return

    # Step 0: Inspect first scene if requested or if SAR dtype unknown
    if args.inspect_only or not any(s.get("sar_dtype") for s in [{"dummy": 0}]):
        # Find first train spring scene to inspect
        first_scene = None
        for split in ["train"]:
            for season in seasons:
                scenes = SCENE_MANIFEST[split].get(season, [])
                if scenes:
                    first_scene = (split, season, scenes[0])
                    break
            if first_scene:
                break

        if first_scene:
            split, season, scene_num = first_scene
            parquet_name = f"scene_{scene_num}.parquet"
            parquet_url = f"{HF_BASE_URL}/{season}/{parquet_name}"
            parquet_path = raw_dir / season / parquet_name

            if not parquet_path.exists():
                print(f"[INSPECT] Downloading first scene for range inspection...")
                download_parquet(parquet_url, parquet_path)

            stats, n_rows = inspect_parquet(parquet_path)
            print(f"[INSPECT] {split}/{season}/scene_{scene_num}: {n_rows} rows")
            print(f"  SAR: shape={stats['sar_shape']}, min={stats['sar_min']:.3f}, max={stats['sar_max']:.3f}, mean={stats['sar_mean']:.3f}")
            print(f"  Optical: shape={stats['opt_shape']}, cloudy=[{stats['cloudy_min']}, {stats['cloudy_max']}], target=[{stats['target_min']}, {stats['target_max']}]")

            # Decide SAR storage dtype
            if stats['sar_min'] >= 0 and stats['sar_max'] <= 255:
                sar_dtype = np.uint8
                print(f"  -> SAR appears to be 0-255 uint8; will store as uint8")
            else:
                sar_dtype = np.float32
                print(f"  -> SAR range outside 0-255; will store as float32")

            if args.inspect_only:
                return
        else:
            sar_dtype = np.float32
    else:
        sar_dtype = np.float32

    # Parallel processing
    print(f"\nProcessing {len(work)} scenes with {args.workers} workers...")
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                process_scene, split, season, scene_num,
                raw_dir, out_dir, sar_dtype, progress, manifest, args.dry_run
            ): (split, season, scene_num)
            for split, season, scene_num in work
        }

        for future in as_completed(futures):
            split, season, scene_num = futures[future]
            try:
                future.result()
            except Exception as e:
                scene_key = f"{split}/{season}/scene_{scene_num}"
                print(f"  [EXCEPTION] {scene_key}: {e}")
                progress["failed"].append(scene_key)
                save_progress(raw_dir, progress)

    print("\n=== SUMMARY ===")
    print(f"Completed: {len(progress['completed'])} scenes")
    print(f"Failed: {len(progress['failed'])} scenes")
    for split in ["train", "val", "test"]:
        total_patches = sum(e["n_patches"] for e in manifest.get(split, []))
        print(f"  {split}: {len(manifest.get(split, []))} scenes, {total_patches} patches")

    if progress["failed"]:
        print(f"\nFailed scenes: {progress['failed']}")
        sys.exit(1)


if __name__ == "__main__":
    main()