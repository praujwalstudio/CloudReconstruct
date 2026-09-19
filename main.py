"""
CloudReconstruct — Adaptive Multi-Source Cloud Removal for LISS-IV Imagery
======================================================================
Main entry point. Orchestrates data pipeline, automated demo verification, and SOTA accuracy benchmarking.

Usage:
    python main.py                  Run full pipeline
    python main.py --demo           Run automated demonstration on 3 reference scenes
    python main.py --step benchmark Run official SOTA accuracy benchmarking
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
import rasterio

sys.path.insert(0, str(Path(__file__).parent))

from src.config import (
    RAW, LISS4_RAW, ALIGNED, CLOUD_MASKS, CLOUD_FREE, GEOTIFF_OUT, OUTPUTS, CHECKPOINTS
)
from src.data.band_harmonization import harmonize_s2_to_liss4
from src.preprocessing.download_data import list_available_scenes
from src.preprocessing.align import align_all_scenes
from src.preprocessing.cloud_mask import process_all
from src.preprocessing.patch_generator import PatchGenerator
from src.evaluation.inference import CloudFreeInference
from src.evaluation.geotiff_output import write_analysis_ready_product
from src.evaluation.report_generator import QualityReportGenerator
from src.evaluation.benchmark_accuracy import run_benchmark as run_sota_benchmark


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
    print("\n" + "=" * 60)
    print("STEP 5: Cloud-Free Inference")
    print("=" * 60)

    scenes = sorted(ALIGNED.glob("*.tif")) + sorted(ALIGNED.glob("*.tiff"))
    if not scenes:
        scenes = sorted(LISS4_RAW.glob("*.tif"))
    if not scenes:
        print("[ERROR] No scenes found. Run download/align steps first.")
        return []

    model = CloudFreeInference(
        device="cpu",
        density_ckpt=CHECKPOINTS / "density_model" / "best_model.pth",
        correction_ckpt=CHECKPOINTS / "correction_model" / "best_model.pth",
        sar_ckpt=CHECKPOINTS / "diffusion_model" / "best_model.pth",
        temporal_ckpt=CHECKPOINTS / "temporal_model" / "best_model.pth",
    )
    CLOUD_FREE.mkdir(parents=True, exist_ok=True)
    GEOTIFF_OUT.mkdir(parents=True, exist_ok=True)

    results = []
    for scene in scenes:
        print(f"\n  Processing: {scene.name} ...", end=" ")
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


def step_benchmark():
    print("\n" + "=" * 60)
    print("STEP 6: Official SOTA Accuracy Benchmarking")
    print("=" * 60)
    multi_res, mono_res, report = run_sota_benchmark(device="cpu", output_dir=OUTPUTS)
    print("\n" + report)
    return {"multi_temporal": multi_res, "mono_temporal": mono_res}


def run_demo():
    """Executes automated end-to-end demo on 3 reference scenes (Thin, Medium, Dense clouds)."""
    print("\n" + "=" * 70)
    print("CloudReconstruct — Automated Reference Demonstration Mode")
    print("=" * 70)

    demo_output_geotiff = OUTPUTS / "demo" / "geotiff"
    demo_output_reports = OUTPUTS / "demo" / "reports"
    demo_output_geotiff.mkdir(parents=True, exist_ok=True)
    demo_output_reports.mkdir(parents=True, exist_ok=True)

    cloudy_scenes = sorted((RAW / "cloudy").glob("*.tif"))
    if not cloudy_scenes:
        print("[ERROR] No raw reference scenes found in data/raw/cloudy.")
        return

    demo_scenes = cloudy_scenes[:3]
    labels = ["Thin Cloud Condition", "Medium Cloud Condition", "Dense Cloud Condition"]

    model = CloudFreeInference(
        device="cpu",
        density_ckpt=CHECKPOINTS / "density_model" / "best_model.pth",
        correction_ckpt=CHECKPOINTS / "correction_model" / "best_model.pth",
        sar_ckpt=CHECKPOINTS / "diffusion_model" / "best_model.pth",
        temporal_ckpt=CHECKPOINTS / "temporal_model" / "best_model.pth",
    )
    report_gen = QualityReportGenerator(output_dir=demo_output_reports)

    print(f"[*] Initialized model pipeline with active checkpoints.")
    print(f"[*] Processing {len(demo_scenes)} benchmark reference scenes...\n")

    summary_cards = []

    for i, scene_path in enumerate(demo_scenes):
        tag = labels[i] if i < len(labels) else f"Reference Scene {i+1}"
        base_name = scene_path.name.replace("cloudy_", "")
        print(f"--- [{i+1}/{len(demo_scenes)}] Processing: {scene_path.name} ({tag}) ---")

        with rasterio.open(scene_path) as src:
            raw_s2 = src.read()
            profile = src.profile.copy()

        # Harmonize to 3 bands (G, R, NIR)
        cloudy_3b = harmonize_s2_to_liss4(raw_s2, scale_toa=True)
        cloudy_hwc = np.moveaxis(cloudy_3b, 0, -1)

        # Look for paired SAR & DEM
        sar_data, dem_data = None, None
        sar_cand = RAW / "sigma0" / f"sigma0_{base_name}"
        if sar_cand.exists():
            with rasterio.open(sar_cand) as src_s1:
                sar_data = np.moveaxis(src_s1.read()[:2], 0, -1)

        dem_cand = RAW / "dem" / f"dem_{base_name}"
        if dem_cand.exists():
            with rasterio.open(dem_cand) as src_dem:
                dem_data = src_dem.read(1)

        dem_processor = None
        if dem_data is not None:
            from src.evaluation.dem_integration import TerrainProcessor
            dem_processor = TerrainProcessor(resolution=5.8)
            dem_processor.load_from_array(dem_data)

        # Run inference
        res = model.correct(cloudy_hwc, sar=sar_data, dem_processor=dem_processor, data_max=1.0)
        corrected = res["corrected"]
        density = res["density"]
        confidence = res["confidence"]
        ars = res["ars"]
        grade = model.readiness.grade(ars["ars"])

        # 1. Export Analysis-Ready GeoTIFF
        out_tif = demo_output_geotiff / f"cloud_free_{scene_path.name}"
        write_analysis_ready_product(
            out_tif,
            corrected,
            confidence_map=confidence,
            ars_result={"ars": ars["ars"], "grade": grade, "components": ars.get("components", {})},
            profile=profile,
        )

        # 2. Export Quality Inspection PDF & JSON
        rep = report_gen.generate_report(
            image_id=scene_path.stem,
            density_map=density,
            confidence_map=confidence,
            corrected_image=corrected,
            cloudy_image=cloudy_hwc,
            ars_result={"ars": ars["ars"], "grade": grade, "components": ars.get("components", {})},
        )

        summary_cards.append({
            "scene": scene_path.name,
            "condition": tag,
            "ars": ars["ars"],
            "grade": grade,
            "cloud_cover": rep["cloud_cover_pct"],
            "recovered_area": rep["recovered_surface_area_pct"],
            "confidence": rep["average_confidence_pct"],
            "geotiff": str(out_tif),
            "pdf_report": rep["pdf_report_path"],
        })

        print(f"    -> ARS Score: {ars['ars']:.4f} (Grade: {grade})")
        print(f"    -> Cloud Cover: {rep['cloud_cover_pct']:.1f}% | Confidence: {rep['average_confidence_pct']:.1f}%")
        print(f"    -> GeoTIFF: {out_tif.name}")
        print(f"    -> PDF QA Report: {Path(rep['pdf_report_path']).name}\n")

    print("=" * 70)
    print("DEMO EXECUTION SUMMARY")
    print("=" * 70)
    print(f"{'Scene Name':<30} | {'Condition':<20} | {'ARS':<7} | {'Grade':<5} | {'Confidence':<10}")
    print("-" * 80)
    for c in summary_cards:
        print(f"{c['scene'][:30]:<30} | {c['condition']:<20} | {c['ars']:.4f}  | {c['grade']:<5} | {c['confidence']:.1f}%")
    print("=" * 70)
    print(f"[OK] GeoTIFFs saved to: {demo_output_geotiff}")
    print(f"[OK] QA PDF Reports saved to: {demo_output_reports}")


def infer_single_image(input_path: Path, output_path: Path = None, device: str = "cpu"):
    """Performs cloud removal on a single standalone optical image (LISS-IV, Sentinel-2, PNG, GeoTIFF)."""
    input_path = Path(input_path)
    if not input_path.exists():
        print(f"[ERROR] Input file does not exist: {input_path}")
        return None

    print(f"\n[INFO] Processing single satellite image: {input_path.name}")
    profile = None
    if input_path.suffix.lower() in (".tif", ".tiff"):
        with rasterio.open(input_path) as src:
            image = src.read()
            profile = src.profile.copy()
        if image.ndim == 3:
            if image.shape[0] == 13:
                image = harmonize_s2_to_liss4(image, scale_toa=True)
                if image.ndim == 3:
                    image = np.moveaxis(image, 0, -1)
            else:
                image = np.moveaxis(image, 0, -1)
        elif image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
    else:
        from PIL import Image
        pil_img = Image.open(input_path).convert("RGB")
        image = np.array(pil_img, dtype=np.uint8)

    model = CloudFreeInference(
        device=device,
        density_ckpt=CHECKPOINTS / "density_model" / "best_model.pth",
        correction_ckpt=CHECKPOINTS / "correction_model" / "best_model.pth",
        sar_ckpt=CHECKPOINTS / "diffusion_model" / "best_model.pth",
        temporal_ckpt=CHECKPOINTS / "temporal_model" / "best_model.pth",
    )

    result = model.correct(image, data_max=1.0 if image.max() <= 1.0 else 65535.0 if image.dtype == np.uint16 else 255.0)

    if output_path is None:
        GEOTIFF_OUT.mkdir(parents=True, exist_ok=True)
        if profile is not None:
            output_path = GEOTIFF_OUT / f"cloud_free_{input_path.stem}.tif"
        else:
            output_path = GEOTIFF_OUT / f"cloud_free_{input_path.stem}.png"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if profile is not None and output_path.suffix.lower() in (".tif", ".tiff"):
        write_analysis_ready_product(
            output_path, result["corrected"], result["confidence"],
            result["ars"], profile
        )
    else:
        from PIL import Image
        corr_arr = result["corrected"]
        if corr_arr.dtype == np.uint16:
            corr_u8 = (corr_arr / 256.0).clip(0, 255).astype(np.uint8)
        else:
            corr_u8 = corr_arr.astype(np.uint8)
        Image.fromarray(corr_u8).save(output_path)

    grade = model.readiness.grade(result["ars"]["ars"])
    print(f"\n[OK] Single Image Cloud Removal Completed Successfully!")
    print(f"  - Output Saved: {output_path}")
    print(f"  - ARS Score:    {result['ars']['ars']:.4f} (Grade: {grade})")
    print(f"  - Mean Conf:    {result['confidence'].mean() * 100:.2f}%\n")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="CloudReconstruct Data Pipeline")
    parser.add_argument("--demo", action="store_true", help="Run automated demonstration on reference scenes")
    parser.add_argument("--infer-single", type=str, default=None,
                        help="Path to a single optical image (LISS-IV, Sentinel-2, PNG, GeoTIFF) to process")
    parser.add_argument("--output", type=str, default=None,
                        help="Optional output path for single image inference")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Inference device (cuda or cpu)")
    parser.add_argument("--step", type=str, default="all",
                        choices=["all", "download", "align", "mask", "patch", "infer", "benchmark"],
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

    if args.infer_single:
        infer_single_image(Path(args.infer_single), args.output, device=args.device)
        return

    if args.demo:
        run_demo()
        return

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
    elif args.step == "benchmark":
        step_benchmark()
    else:
        globals()[f"step_{args.step}"]()

    print("\n" + "=" * 60)
    print("Pipeline complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()

