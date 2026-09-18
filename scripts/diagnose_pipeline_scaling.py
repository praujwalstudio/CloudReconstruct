"""Data Normalization and Pipeline Scaling Diagnostics Tool
=========================================================
Inspects tensor values, range distributions, and per-module outputs
across the data ingestion layer and the multi-tier neural network.
"""

import sys
from pathlib import Path
import numpy as np
import rasterio
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RAW, CHECKPOINTS
from src.data.band_harmonization import harmonize_s2_to_liss4
from src.data.sen12ms_dataset import normalize_sar
from src.evaluation.inference import CloudFreeInference, _numpy_to_tensor
from src.evaluation.metrics import compute_all_metrics


def print_stats(name: str, arr_or_tensor):
    if isinstance(arr_or_tensor, torch.Tensor):
        a = arr_or_tensor.detach().cpu().float().numpy()
    else:
        a = np.array(arr_or_tensor, dtype=np.float32)
    print(f"  [{name:<25}] shape={str(a.shape):<18} min={a.min():9.4f}  max={a.max():9.4f}  mean={a.mean():9.4f}  std={a.std():9.4f}")


def run_diagnostics():
    print("=" * 80)
    print("CloudReconstruct — Data Normalization & Tensor Scaling Diagnostics")
    print("=" * 80)

    cloudy_files = sorted((RAW / "cloudy").glob("*.tif"))
    if not cloudy_files:
        print("[ERROR] No raw files found in data/raw/cloudy.")
        return

    sample_scene = cloudy_files[0]
    base_name = sample_scene.name.replace("cloudy_", "")
    clear_scene = RAW / "clear" / f"clear_{base_name}"
    sar_scene = RAW / "sigma0" / f"sigma0_{base_name}"
    dem_scene = RAW / "dem" / f"dem_{base_name}"

    print(f"\n[1] Inspecting Raw File Data ({sample_scene.name}):")
    with rasterio.open(sample_scene) as src:
        raw_cloudy = src.read()
        print_stats("Raw S2 Cloudy", raw_cloudy)

    with rasterio.open(clear_scene) as src:
        raw_clear = src.read()
        print_stats("Raw S2 Clear (Target)", raw_clear)

    if sar_scene.exists():
        with rasterio.open(sar_scene) as src:
            raw_sar = src.read()
            print_stats("Raw SAR (dB)", raw_sar)

    if dem_scene.exists():
        with rasterio.open(dem_scene) as src:
            raw_dem = src.read()
            print_stats("Raw DEM (meters)", raw_dem)

    print("\n[2] Ingestion & Harmonization Transforms:")
    cloudy_3b = harmonize_s2_to_liss4(raw_cloudy, scale_toa=True)
    clear_3b = harmonize_s2_to_liss4(raw_clear, scale_toa=True)
    cloudy_hwc = np.moveaxis(cloudy_3b, 0, -1)
    clear_hwc = np.moveaxis(clear_3b, 0, -1)
    print_stats("Harmonized Cloudy (0-1)", cloudy_hwc)
    print_stats("Harmonized Clear (0-1)", clear_hwc)

    sar_hwc = None
    if sar_scene.exists():
        sar_hwc = np.moveaxis(raw_sar[:2], 0, -1)
        sar_norm = normalize_sar(sar_hwc)
        print_stats("Normalized SAR (-1 to 1)", sar_norm)

    print("\n[3] Model Pipeline Intermediate Tensor Scaling:")
    model = CloudFreeInference(device="cpu")
    model.pipeline.eval()

    in_tensor = _numpy_to_tensor(cloudy_hwc, "cpu", data_max=1.0)
    ref_tensor = _numpy_to_tensor(clear_hwc, "cpu", data_max=1.0)
    sar_tensor = _numpy_to_tensor(sar_norm, "cpu") if sar_hwc is not None else None

    print_stats("Input Tensor (liss4)", in_tensor)
    if sar_tensor is not None:
        print_stats("Input Tensor (sar)", sar_tensor)

    with torch.no_grad():
        density = model.pipeline.density_net(in_tensor)
        print_stats("CloudDensityNet Output", density)

        thin_out = model.pipeline.correction_net(in_tensor, density)
        print_stats("ThinCorrection Output", thin_out)

        temp_out = model.pipeline.temporal_fusion(in_tensor, ref_tensor, density)
        print_stats("TemporalFusion Output", temp_out)

        if sar_tensor is not None:
            sar_out = model.pipeline.sar_fusion(in_tensor, sar_tensor[:, :2])
            print_stats("SARDiffusion Output", sar_out)

        weights = model.pipeline._compute_blend_weights(density)
        print_stats("Blend Weights (Thin)", weights[:, 0])
        print_stats("Blend Weights (Med)", weights[:, 1])
        print_stats("Blend Weights (Dense)", weights[:, 2])

        final_out, _, _ = model.pipeline(in_tensor, sar_tensor[:, :2] if sar_tensor is not None else None, [ref_tensor])
        print_stats("Final Pipeline Output", final_out)

    print("\n[4] Metric Comparison (Reconstructed vs Clear Target):")
    recon_np = final_out.squeeze(0).permute(1, 2, 0).numpy()
    metrics = compute_all_metrics(recon_np, clear_hwc, data_range=1.0)
    for k, v in metrics.items():
        print(f"  {k.upper():<18}: {v:.4f}")

    print("\n[5] Cloud-Masked Ground-Truth Comparison (Input Cloudy vs Target):")
    base_metrics = compute_all_metrics(cloudy_hwc, clear_hwc, data_range=1.0)
    for k, v in base_metrics.items():
        print(f"  Raw Cloudy {k.upper():<8}: {v:.4f}")

    print("\n" + "=" * 80)
    print("Diagnostics complete.")
    print("=" * 80)


if __name__ == "__main__":
    run_diagnostics()
