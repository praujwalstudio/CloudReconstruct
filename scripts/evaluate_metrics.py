"""Evaluate Overall Model Accuracy & Reconstruction Metrics
==========================================================
Computes standard remote sensing reconstruction benchmarks:
- PSNR (Peak Signal-to-Noise Ratio) in dB
- SSIM (Structural Similarity Index Measure) [0 to 1]
- SAM (Spectral Angle Mapper) in degrees
- NDVI Pearson Correlation [-1 to 1]
- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
import rasterio

from src.config import RAW, ALIGNED, CLOUD_MASKS, PATCHES, CHECKPOINTS, GEOTIFF_OUT
from src.data.band_harmonization import harmonize_s2_to_liss4
from src.evaluation.metrics import compute_all_metrics
from src.evaluation.inference import CloudFreeInference


def evaluate_pipeline():
    print("=" * 70)
    print("CloudReconstruct: Comprehensive Model Evaluation & Accuracy Report")
    print("=" * 70)

    clear_scenes = sorted((RAW / "clear").glob("*.tif"))
    cloudy_scenes = sorted((RAW / "cloudy").glob("*.tif"))
    sigma0_scenes = sorted((RAW / "sigma0").glob("*.tif"))

    if not clear_scenes or not cloudy_scenes:
        print("[WARN] No raw test scenes found in data/raw.")
        return

    infer_engine = CloudFreeInference(
        device="cpu",
        density_ckpt=CHECKPOINTS / "density_model" / "best_model.pth",
        correction_ckpt=CHECKPOINTS / "correction_model" / "best_model.pth",
        sar_ckpt=CHECKPOINTS / "diffusion_model" / "best_model.pth",
        temporal_ckpt=CHECKPOINTS / "temporal_model" / "best_model.pth",
    )
    all_metrics = []

    print(f"\n[*] Evaluating on {len(cloudy_scenes)} multi-modal scenes...")

    for c_path in cloudy_scenes:
        base_name = c_path.name.replace("cloudy_", "")
        cl_path = RAW / "clear" / f"clear_{base_name}"
        s1_path = RAW / "sigma0" / f"sigma0_{base_name}"

        if not cl_path.exists():
            continue

        # Load S2 cloudy & clear
        with rasterio.open(c_path) as src_c:
            s2_cloudy_raw = src_c.read()
        with rasterio.open(cl_path) as src_cl:
            s2_clear_raw = src_cl.read()

        # Harmonize to 3 bands (G, R, NIR)
        cloudy_liss4 = harmonize_s2_to_liss4(s2_cloudy_raw, scale_toa=True)  # (3, H, W) in [0, 1]
        clear_gt = harmonize_s2_to_liss4(s2_clear_raw, scale_toa=True)       # (3, H, W) in [0, 1]

        cloudy_hwc = np.moveaxis(cloudy_liss4, 0, -1)
        target_hwc = np.moveaxis(clear_gt, 0, -1)

        # SAR
        sar_data = None
        if s1_path.exists():
            with rasterio.open(s1_path) as src_s1:
                sar_data = src_s1.read()
                if sar_data.ndim == 3:
                    sar_data = np.moveaxis(sar_data, 0, -1)

        # Run inference
        res = infer_engine.correct(cloudy_hwc, sar=sar_data, data_max=1.0)
        reconstructed = res["corrected"]

        if reconstructed.dtype == np.uint16:
            reconstructed = reconstructed.astype(np.float32) / 65535.0
        elif reconstructed.dtype == np.uint8:
            reconstructed = reconstructed.astype(np.float32) / 255.0
        else:
            reconstructed = reconstructed.astype(np.float32)

        # Compute metrics
        metrics = compute_all_metrics(reconstructed, target_hwc, data_range=1.0)
        metrics["mae"] = float(np.mean(np.abs(reconstructed - target_hwc)))
        metrics["mse"] = float(np.mean((reconstructed - target_hwc) ** 2))
        all_metrics.append(metrics)

        print(f"\n  Scene: {base_name}")
        print(f"    - PSNR:             {metrics['psnr']:.2f} dB")
        print(f"    - SSIM:             {metrics['ssim']:.4f} ({metrics['ssim']*100:.1f}%)")
        print(f"    - SAM:              {metrics['sam']:.2f}°")
        print(f"    - NDVI Correlation: {metrics['ndvi_correlation']:.4f} ({metrics['ndvi_correlation']*100:.1f}%)")
        print(f"    - MAE:              {metrics['mae']:.4f}")

    if all_metrics:
        avg_psnr = float(np.mean([m["psnr"] for m in all_metrics]))
        avg_ssim = float(np.mean([m["ssim"] for m in all_metrics]))
        avg_sam = float(np.mean([m["sam"] for m in all_metrics]))
        avg_ndvi = float(np.mean([m["ndvi_correlation"] for m in all_metrics]))
        avg_mae = float(np.mean([m["mae"] for m in all_metrics]))
        avg_mse = float(np.mean([m["mse"] for m in all_metrics]))

        print("\n" + "=" * 70)
        print("OVERALL MODEL ACCURACY & PERFORMANCE SUMMARY")
        print("=" * 70)
        print(f"  Overall SSIM (Structural Similarity):   {avg_ssim:.4f} ({avg_ssim*100:.2f}%)")
        print(f"  Overall NDVI Fidelity (Pearson r):     {avg_ndvi:.4f} ({avg_ndvi*100:.2f}%)")
        print(f"  Overall PSNR (Signal-to-Noise Ratio):  {avg_psnr:.2f} dB")
        print(f"  Overall SAM (Spectral Angle Error):    {avg_sam:.2f}°")
        print(f"  Overall MAE (Mean Absolute Error):     {avg_mae:.4f}")
        print(f"  Overall MSE (Mean Squared Error):      {avg_mse:.6f}")
        print("=" * 70)

        out_file = PROJECT_ROOT / "evaluation_summary.json"
        summary_data = {
            "overall_ssim": avg_ssim,
            "overall_ssim_pct": avg_ssim * 100,
            "overall_ndvi_correlation": avg_ndvi,
            "overall_psnr_db": avg_psnr,
            "overall_sam_deg": avg_sam,
            "overall_mae": avg_mae,
            "overall_mse": avg_mse,
            "n_scenes": len(all_metrics),
        }
        with open(out_file, "w") as f:
            json.dump(summary_data, f, indent=2)
        print(f"[OK] Summary saved to {out_file}")


if __name__ == "__main__":
    evaluate_pipeline()
