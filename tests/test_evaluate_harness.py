"""Unit Tests for Evaluation & Benchmarking Harness"""

import json
from pathlib import Path
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from rasterio.crs import CRS

from src.evaluation.evaluate import BenchmarkEvaluator, PUBLISHED_BASELINES


class TestBenchmarkEvaluator:
    @pytest.fixture
    def test_scenes(self, tmp_path):
        cloudy_dir = tmp_path / "cloudy"
        clear_dir = tmp_path / "clear"
        sigma0_dir = tmp_path / "sigma0"
        dem_dir = tmp_path / "dem"
        for d in (cloudy_dir, clear_dir, sigma0_dir, dem_dir):
            d.mkdir(parents=True, exist_ok=True)

        h, w = 128, 128
        profile = {
            "driver": "GTiff",
            "height": h,
            "width": w,
            "count": 3,
            "dtype": "uint16",
            "crs": CRS.from_epsg(32645),
            "transform": from_origin(100.0, 100.0, 10.0, 10.0),
        }

        # Write 2 test scenes
        for i in range(2):
            scene_name = f"scene_{i+1}.tif"
            img_cloudy = np.ones((3, h, w), dtype=np.uint16) * 4000
            img_clear = np.ones((3, h, w), dtype=np.uint16) * 1500

            with rasterio.open(cloudy_dir / f"cloudy_{scene_name}", "w", **profile) as dst:
                dst.write(img_cloudy)
            with rasterio.open(clear_dir / f"clear_{scene_name}", "w", **profile) as dst:
                dst.write(img_clear)

            p_s1 = profile.copy()
            p_s1.update(count=2, dtype="float32")
            with rasterio.open(sigma0_dir / f"sigma0_{scene_name}", "w", **p_s1) as dst:
                dst.write(np.zeros((2, h, w), dtype=np.float32) - 15.0)

            p_dem = profile.copy()
            p_dem.update(count=1, dtype="float32")
            with rasterio.open(dem_dir / f"dem_{scene_name}", "w", **p_dem) as dst:
                dst.write(np.ones((h, w), dtype=np.float32) * 500.0, 1)

        return tmp_path

    def test_evaluate_scene(self, test_scenes):
        evaluator = BenchmarkEvaluator(device="cpu")
        cloudy_path = test_scenes / "cloudy" / "cloudy_scene_1.tif"
        clear_path = test_scenes / "clear" / "clear_scene_1.tif"
        s1_path = test_scenes / "sigma0" / "sigma0_scene_1.tif"
        dem_path = test_scenes / "dem" / "dem_scene_1.tif"

        metrics = evaluator.evaluate_scene(cloudy_path, clear_path, s1_path, dem_path)
        assert "psnr" in metrics
        assert "ssim" in metrics
        assert "sam" in metrics
        assert "mae" in metrics
        assert not np.isnan(metrics["psnr"])
        assert not np.isnan(metrics["ssim"])

    def test_run_benchmark(self, test_scenes):
        evaluator = BenchmarkEvaluator(device="cpu")
        results = evaluator.run_benchmark(test_scenes)
        assert "psnr" in results
        assert "ssim" in results
        assert results["num_scenes"] == 2
        assert len(results["scenes"]) == 2

    def test_format_markdown_table(self):
        evaluator = BenchmarkEvaluator(device="cpu")
        sample_results = {
            "psnr": 29.85,
            "ssim": 0.8812,
            "sam": 7.45,
            "mae": 0.0395,
            "num_scenes": 5,
        }
        table = evaluator.format_markdown_table(sample_results)
        assert "DSen2-CR" in table
        assert "GLF-CR" in table
        assert "CloudReconstruct (Ours)" in table
        assert "29.85" in table
        assert "0.8812" in table
