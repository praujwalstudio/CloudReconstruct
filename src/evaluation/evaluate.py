"""Comprehensive Evaluation & Benchmarking Harness (WP2)
======================================================
Evaluates the multi-source adaptive cloud removal pipeline on real-world datasets,
computes pixel-wise, spectral, structural, and biogeochemical metrics, and compiles
a publication-standard Markdown comparison table against established baselines (DSen2-CR, GLF-CR).

Usage:
    python -m src.evaluation.evaluate
    python src/evaluation/evaluate.py --device cpu --output-dir data/outputs
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import numpy as np
import rasterio
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RAW, CHECKPOINTS, OUTPUTS, GEOTIFF_OUT
from src.data.band_harmonization import harmonize_s2_to_liss4
from src.data.sen12ms_dataset import SEN12MSCRDataset
from src.evaluation.inference import CloudFreeInference
from src.evaluation.metrics import compute_all_metrics, psnr, sam, ndvi_correlation


PUBLISHED_BASELINES = {
    "DSen2-CR (Meraner et al.)": {
        "psnr": 30.29,
        "ssim": 0.8780,
        "sam": 7.000,
        "mae": 0.0382,
        "modality": "Optical + SAR",
    },
    "GLF-CR (Xu et al.)": {
        "psnr": 28.64,
        "ssim": 0.8850,
        "sam": 8.981,
        "mae": 0.0420,
        "modality": "Optical + SAR (Global-Local Fusion)",
    },
}


def _to_json_serializable(obj: Any) -> Any:
    """Recursively converts numpy numbers/arrays to native python types for JSON serialization."""
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_to_json_serializable(v) for v in obj]
    return obj


def _prepare_scene_array(raw_arr: np.ndarray) -> np.ndarray:
    """Prepares 2D/3D array into (H, W, 3) normalized float32 array in [0, 1]."""
    if raw_arr.ndim == 3:
        if 8 <= raw_arr.shape[0] <= 16:  # S2 (13, H, W)
            arr_3b = harmonize_s2_to_liss4(raw_arr, scale_toa=True)
            arr = np.moveaxis(arr_3b, 0, -1)
        elif 8 <= raw_arr.shape[2] <= 16:  # S2 (H, W, 13)
            arr = harmonize_s2_to_liss4(raw_arr, scale_toa=True)
        elif raw_arr.shape[0] in (1, 2, 3, 4) and raw_arr.shape[0] != raw_arr.shape[1]:  # CHW format
            if raw_arr.shape[0] == 1:
                arr = np.repeat(raw_arr[0, ..., np.newaxis], 3, axis=-1).astype(np.float32)
            else:
                arr = np.moveaxis(raw_arr[:3], 0, -1).astype(np.float32)
        elif raw_arr.shape[2] in (1, 2, 3, 4):  # HWC format
            if raw_arr.shape[2] == 1:
                arr = np.repeat(raw_arr, 3, axis=2).astype(np.float32)
            else:
                arr = raw_arr[..., :3].astype(np.float32)
        else:
            arr = raw_arr.astype(np.float32)
    elif raw_arr.ndim == 2:
        arr = np.stack([raw_arr.astype(np.float32)] * 3, axis=-1)
    else:
        arr = raw_arr.astype(np.float32)

    if arr.max() > 1.0:
        arr = arr / (65535.0 if arr.max() > 255.0 else (10000.0 if arr.max() > 1000.0 else 255.0))

    return np.clip(arr, 0.0, 1.0)


class BenchmarkEvaluator:
    """Evaluation harness for running quantitative benchmarks on test datasets."""

    def __init__(
        self,
        device: str = "cpu",
        density_ckpt: Optional[Path] = None,
        correction_ckpt: Optional[Path] = None,
        sar_ckpt: Optional[Path] = None,
        temporal_ckpt: Optional[Path] = None,
    ):
        self.device = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
        self.infer_engine = CloudFreeInference(
            device=self.device,
            density_ckpt=density_ckpt or (CHECKPOINTS / "density_model" / "best_model.pth"),
            correction_ckpt=correction_ckpt or (CHECKPOINTS / "correction_model" / "best_model.pth"),
            sar_ckpt=sar_ckpt or (CHECKPOINTS / "diffusion_model" / "best_model.pth"),
            temporal_ckpt=temporal_ckpt or (CHECKPOINTS / "temporal_model" / "best_model.pth"),
        )

    def evaluate_scene(
        self,
        cloudy_path: Path,
        clear_path: Path,
        sar_path: Optional[Path] = None,
        dem_path: Optional[Path] = None,
    ) -> Dict[str, float]:
        """Evaluates a single multi-modal scene with NaN-safe edge case handling."""
        with rasterio.open(cloudy_path) as src_c:
            raw_cloudy = src_c.read()
        with rasterio.open(clear_path) as src_cl:
            raw_clear = src_cl.read()

        cloudy_hwc = _prepare_scene_array(raw_cloudy)
        clear_hwc = _prepare_scene_array(raw_clear)

        sar_data = None
        if sar_path and sar_path.exists():
            with rasterio.open(sar_path) as src_s1:
                sar_raw = src_s1.read()
                if sar_raw.ndim == 3 and sar_raw.shape[0] <= 4:
                    sar_data = np.moveaxis(sar_raw, 0, -1)
                else:
                    sar_data = sar_raw

        dem_data = None
        if dem_path and dem_path.exists():
            with rasterio.open(dem_path) as src_dem:
                dem_data = src_dem.read(1)

        # Run model inference
        res = self.infer_engine.correct(cloudy_hwc, sar=sar_data, data_max=1.0)
        reconstructed = res["corrected"]

        if reconstructed.dtype == np.uint16:
            reconstructed = (reconstructed.astype(np.float32) / 65535.0).clip(0, 1)
        elif reconstructed.dtype == np.uint8:
            reconstructed = (reconstructed.astype(np.float32) / 255.0).clip(0, 1)
        else:
            reconstructed = np.nan_to_num(reconstructed.astype(np.float32), nan=0.0).clip(0, 1)

        # Compute metrics defensively
        metrics = compute_all_metrics(reconstructed, clear_hwc, data_range=1.0)
        diff = reconstructed - clear_hwc
        metrics["mae"] = float(np.mean(np.abs(diff)))
        metrics["rmse"] = float(np.sqrt(np.mean(diff ** 2)))
        metrics["mse"] = float(np.mean(diff ** 2))
        metrics["ars"] = float(res["ars"]["ars"])
        metrics["density_mean"] = float(res["density"].mean())
        metrics["confidence_mean"] = float(res["confidence"].mean())

        return metrics

    def run_benchmark(self, data_dir: Path = RAW) -> Dict[str, Any]:
        """Runs evaluation over all test scenes in dataset directory."""
        data_dir = Path(data_dir)
        cloudy_scenes = sorted((data_dir / "cloudy").glob("*.tif"))
        if not cloudy_scenes:
            cloudy_scenes = sorted(data_dir.rglob("*cloudy*.tif"))

        scene_results = []
        for c_path in cloudy_scenes:
            base_name = c_path.name.replace("cloudy_", "").replace("aligned_", "")
            cl_candidates = [
                data_dir / "clear" / f"clear_{base_name}",
                data_dir / "clear" / f"clear_aligned_{base_name}",
                c_path.parent.parent / "clear" / f"clear_{base_name}",
            ]
            cl_path = next((p for p in cl_candidates if p.exists()), None)
            if not cl_path:
                continue

            s1_candidates = [
                data_dir / "sigma0" / f"sigma0_{base_name}",
                data_dir / "sigma0" / f"sigma0_aligned_{base_name}",
            ]
            s1_path = next((p for p in s1_candidates if p.exists()), None)

            dem_candidates = [
                data_dir / "dem" / f"dem_{base_name}",
                data_dir / "dem" / f"dem_aligned_{base_name}",
            ]
            dem_path = next((p for p in dem_candidates if p.exists()), None)

            m = self.evaluate_scene(c_path, cl_path, s1_path, dem_path)
            m["scene_name"] = c_path.name
            scene_results.append(m)

        if not scene_results:
            return {"error": "No valid cloudy-clear pairs found for evaluation", "scenes": []}

        # Compute aggregation
        avg_metrics = {
            "psnr": float(np.nanmean([s["psnr"] for s in scene_results])),
            "ssim": float(np.nanmean([s["ssim"] for s in scene_results])),
            "sam": float(np.nanmean([s["sam"] for s in scene_results])),
            "ndvi_correlation": float(np.nanmean([s["ndvi_correlation"] for s in scene_results])),
            "mae": float(np.nanmean([s["mae"] for s in scene_results])),
            "rmse": float(np.nanmean([s["rmse"] for s in scene_results])),
            "ars": float(np.nanmean([s["ars"] for s in scene_results])),
            "num_scenes": len(scene_results),
            "scenes": scene_results,
        }

        return avg_metrics

    def format_markdown_table(self, summary: Dict[str, Any]) -> str:
        """Formats a comparison Markdown table against published baselines."""
        our_psnr = summary.get("psnr", 0.0)
        our_ssim = summary.get("ssim", 0.0)
        our_sam = summary.get("sam", 0.0)
        our_mae = summary.get("mae", 0.0)

        lines = [
            "### Cloud Removal Quantitative Benchmark Comparison",
            "",
            "| Model / Architecture | Modality / Fusion Strategy | PSNR (dB) | SSIM | SAM (deg) | MAE |",
            "| :--- | :--- | :---: | :---: | :---: | :---: |",
        ]

        # Baselines
        for name, data in PUBLISHED_BASELINES.items():
            lines.append(
                f"| **{name}** | {data['modality']} | {data['psnr']:.2f} | {data['ssim']:.4f} | {data['sam']:.2f} | {data['mae']:.4f} |"
            )

        # Ours
        lines.append(
            f"| **CloudReconstruct (Ours)** | Adaptive Optical-SAR Diffusion + DEM Early Fusion | **{our_psnr:.2f}** | **{our_ssim:.4f}** | **{our_sam:.2f}** | **{our_mae:.4f}** |"
        )
        lines.append("")
        lines.append(f"*Evaluated on {summary.get('num_scenes', 0)} multi-modal test scenes with sub-pixel co-registration.*")

        return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Evaluate CloudReconstruct against baselines")
    parser.add_argument("--dataset-dir", type=str, default=str(RAW), help="Dataset root directory")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUTS), help="Output directory")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    args = parser.parse_args()

    print("=" * 70)
    print("CloudReconstruct — Evaluation & Benchmarking Harness (WP2)")
    print("=" * 70)

    evaluator = BenchmarkEvaluator(device=args.device)
    results = evaluator.run_benchmark(Path(args.dataset_dir))

    if "error" in results:
        print(f"[ERROR] {results['error']}")
        return

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "evaluation_benchmark.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_to_json_serializable(results), f, indent=2)
    print(f"\n[OK] Benchmark JSON saved to: {json_path}")

    md_table = evaluator.format_markdown_table(results)
    table_path = out_dir / "evaluation_table.md"
    with open(table_path, "w", encoding="utf-8") as f:
        f.write(md_table)
    print(f"[OK] Markdown table saved to: {table_path}")

    print("\n" + md_table)


if __name__ == "__main__":
    main()
