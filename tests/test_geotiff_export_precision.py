"""Unit Tests for Precision-Preserving GeoTIFF Exporter"""

from pathlib import Path
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from rasterio.crs import CRS

from src.evaluation.geotiff_output import (
    write_geotiff,
    write_analysis_ready_product,
    export_analysis_ready_geotiff,
    read_geotiff_analysis,
)


class TestGeoTIFFPrecision:
    def test_precision_preservation_and_crs(self, tmp_path):
        out_path = tmp_path / "test_precision.tif"
        h, w = 128, 128
        img = np.random.uniform(0, 1, (h, w, 3)).astype(np.float32)
        conf = np.random.uniform(0.5, 1.0, (h, w)).astype(np.float32)

        crs = CRS.from_epsg(32645)
        transform = from_origin(500000.0, 3000000.0, 5.8, 5.8)
        profile = {
            "driver": "GTiff",
            "height": h,
            "width": w,
            "crs": crs,
            "transform": transform,
            "nodata": 0.0,
        }

        ars_result = {
            "ars": 0.895,
            "grade": "A",
            "components": {"density": 0.9, "confidence": 0.89},
            "weights": {"density": 0.5, "confidence": 0.5},
        }

        write_analysis_ready_product(out_path, img, confidence_map=conf, ars_result=ars_result, profile=profile)
        assert out_path.exists()

        # Read back and verify spatial accuracy
        read_res = read_geotiff_analysis(out_path)
        assert read_res["bands"] == 4  # 3 image bands + 1 confidence
        assert "32645" in read_res["crs"]
        assert read_res["transform"][0] == 5.8  # pixel width
        assert read_res["ars"] == 0.895
        assert read_res["ars_components"]["density"] == 0.9

    def test_export_analysis_ready_geotiff_with_source(self, tmp_path):
        src_path = tmp_path / "src_scene.tif"
        out_path = tmp_path / "out_scene.tif"
        h, w = 64, 64

        crs = CRS.from_epsg(4326)
        transform = from_origin(77.5, 12.9, 0.0001, 0.0001)
        src_prof = {
            "driver": "GTiff",
            "height": h,
            "width": w,
            "count": 3,
            "dtype": "float32",
            "crs": crs,
            "transform": transform,
        }

        with rasterio.open(src_path, "w", **src_prof) as dst:
            dst.write(np.ones((3, h, w), dtype=np.float32) * 0.5)
            dst.update_tags(satellite="LISS-IV", sensor="Optical")

        img_corrected = np.ones((h, w, 3), dtype=np.float32) * 0.2
        export_analysis_ready_geotiff(
            out_path,
            img_corrected,
            src_geotiff_path=src_path,
            ars_result={"ars": 0.92, "grade": "A"},
        )

        assert out_path.exists()
        read_back = read_geotiff_analysis(out_path)
        assert "4326" in read_back["crs"]
        assert read_back["metadata"].get("satellite") == "LISS-IV"
        assert read_back["ars"] == 0.92
