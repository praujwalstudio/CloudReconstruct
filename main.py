"""
CloudReconstruct — Adaptive Multi-Source Cloud Removal for LISS-IV Imagery
======================================================================
Main entry point. Orchestrates the data pipeline: download → align → mask → patch → infer.

Usage:
    python main.py                  Run full pipeline
    python main.py --step download  Run only download phase
    python main.py --step align     Run only alignment phase
    python main.py --step mask      Run only cloud masking phase
    python main.py --step patch     Run only patching phase
    python main.py --step infer     Run only inference (cloud removal)
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.config import LISS4_RAW
from src.preprocessing.download_data import list_available_scenes
from src.preprocessing.align import align_all_scenes
from src.preprocessing.cloud_mask import process_all
from src.preprocessing.patch_generator import PatchGenerator
from src.config import ALIGNED, CLOUD_MASKS, CLOUD_FREE, GEOTIFF_OUT


def step_download():
    print("\n" + "=" * 60)
    print("STEP 1: Check available data")
    print("=" * 60)
    summary = list_available_scenes()
    total = sum(len(v) for v in summary.values())
    if total == 0:
        print("\n[INFO] No data found. Manual download instructions printed above.")
        print("After downloading, re-run this step to verify.")
    else:
        print(f"\n[OK] {total} total files found across all sources.")
    return summary


def step_align():
    print("\n" + "=" * 60)
    print("STEP 2: Co-registration")
    print("=" * 60)
    results = align_all_scenes()
    success = sum(1 for r in results if r["aligned"])
    print(f"\n[OK] {success}/{len(results)} scenes aligned successfully.")
    return results


def step_mask():
    print("\n" + "=" * 60)
    print("STEP 3: Cloud Masking")
    print("=" * 60)
    results = process_all()
    print(f"\n[OK] {len(results)} scenes masked.")
    return results


def step_patch():
    print("\n" + "=" * 60)
    print("STEP 4: Patch Generation")
    print("=" * 60)
    scenes = sorted(ALIGNED.glob("*.tif")) + sorted(ALIGNED.glob("*.tiff"))
    masks = sorted(CLOUD_MASKS.glob("mask_*.tif"))

    if not scenes:
        scenes = sorted(LISS4_RAW.glob("*.tif"))

    if not masks and scenes:
        print("[WARN] No masks found. Run mask step first or using raw scenes only.")
        return {}

    if not scenes:
        print("[ERROR] No scenes found. Run download/align steps first.")
        return {}

    gen = PatchGenerator()
    summary = gen.generate_dataset(scenes, masks)
    return summary


def step_infer():
    from src.evaluation.inference import CloudFreeInference

    print("\n" + "=" * 60)
    print("STEP 5: Cloud-Free Inference")
    print("=" * 60)

    scenes = sorted(ALIGNED.glob("*.tif")) + sorted(ALIGNED.glob("*.tiff"))
    if not scenes:
        scenes = sorted(LISS4_RAW.glob("*.tif"))
    if not scenes:
        print("[ERROR] No scenes found. Run download/align steps first.")
        return []

    model = CloudFreeInference(device="cpu")
    CLOUD_FREE.mkdir(parents=True, exist_ok=True)
    GEOTIFF_OUT.mkdir(parents=True, exist_ok=True)

    results = []
    for scene in scenes:
        print(f"\n  Processing: {scene.name} ...", end=" ")
        import rasterio
        with rasterio.open(scene) as src:
            image = src.read()
            profile = src.profile
        if image.ndim == 3:
            image = np.moveaxis(image, 0, -1)

        out_name = f"cloud_free_{scene.stem}.tif"
        out_path = GEOTIFF_OUT / out_name
        result_path = model.correct_and_save(out_path, image, profile=profile)
        results.append(str(result_path))
        print("OK")

    print(f"\n[OK] {len(results)}/{len(scenes)} scenes processed.")
    print(f"Output directory: {GEOTIFF_OUT}")
    return results


def main():
    parser = argparse.ArgumentParser(description="CloudReconstruct Data Pipeline")
    parser.add_argument("--step", type=str, default="all",
                        choices=["all", "download", "align", "mask", "patch", "infer"],
                        help="Pipeline step to run")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be done without executing")
    args = parser.parse_args()

    print(r"""
     ___ _                 _    ____                                        _   
    / __| |___  __ _ _ _  | |__|___ \ _____ __ ___ _ _ __ _ _ __  ___ _ _ | |_ 
   | (__| / _ \/ _` | '_| | '_ \ __) / _ \ V  V / '_/ _` | '_ \/ -_) '_||  _|
    \___|_\___/\__,_|_|   |_.__/____/\___/\_/\_/|_| \__,_| .__/\___|_|   \__|
                                                          |_|                  
    Adaptive Multi-Source Cloud Removal for LISS-IV Imagery
    ==================================================================
    """)

    steps = ["download", "align", "mask", "patch", "infer"]

    if args.dry_run:
        print("\n[Dry Run] Steps that would execute:")
        if args.step == "all":
            for s in steps:
                print(f"  - {s}")
        else:
            print(f"  - {args.step}")
        return

    if args.step == "all":
        for s in steps:
            globals()[f"step_{s}"]()
    else:
        globals()[f"step_{args.step}"]()

    print("\n" + "=" * 60)
    print("Pipeline complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
