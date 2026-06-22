import numpy as np
from pathlib import Path


class TestWriteGeoTIFF:
    def test_write_single_band(self, tmp_path):
        from src.evaluation.geotiff_output import write_geotiff
        image = np.random.randint(0, 1023, (30, 30), dtype=np.uint16)
        out_path = tmp_path / "output.tif"
        result = write_geotiff(out_path, image)
        assert result.exists()
        assert result.suffix == ".tif"

    def test_write_multi_band(self, tmp_path):
        from src.evaluation.geotiff_output import write_geotiff
        image = np.random.randint(0, 1023, (30, 30, 3), dtype=np.uint16)
        out_path = tmp_path / "multiband.tif"
        write_geotiff(out_path, image)
        assert out_path.exists()

    def test_write_with_confidence(self, tmp_path):
        from src.evaluation.geotiff_output import write_geotiff
        image = np.random.randint(0, 1023, (20, 20, 3), dtype=np.uint16)
        confidence = np.random.rand(20, 20).astype(np.float32)
        out_path = tmp_path / "with_conf.tif"
        write_geotiff(out_path, image, confidence_map=confidence)
        assert out_path.exists()

    def test_write_with_metadata(self, tmp_path):
        from src.evaluation.geotiff_output import write_geotiff
        image = np.random.randint(0, 1023, (20, 20, 3), dtype=np.uint16)
        meta = {"test_key": "test_value", "processing_date": "2025-01-01"}
        out_path = tmp_path / "meta.tif"
        write_geotiff(out_path, image, metadata=meta)
        assert out_path.exists()

    def test_dtype_preserved(self, tmp_path):
        from src.evaluation.geotiff_output import write_geotiff, read_geotiff_analysis
        image = np.random.randint(0, 1023, (20, 20, 3), dtype=np.uint16)
        out_path = tmp_path / "dtype_test.tif"
        write_geotiff(out_path, image)
        result = read_geotiff_analysis(out_path)
        assert result["image"].dtype == np.uint16
        assert result["image"].shape[-1] == 3


class TestReadGeoTIFF:
    def test_read_back(self, tmp_path):
        from src.evaluation.geotiff_output import write_geotiff, read_geotiff_analysis
        image = np.random.randint(0, 1023, (30, 30, 3), dtype=np.uint16)
        meta = {"source": "test"}
        out_path = tmp_path / "readback.tif"
        write_geotiff(out_path, image, metadata=meta)
        result = read_geotiff_analysis(out_path)
        assert np.allclose(result["image"], image)
        assert result["metadata"]["source"] == "test"

    def test_read_with_confidence(self, tmp_path):
        from src.evaluation.geotiff_output import write_geotiff, read_geotiff_analysis
        image = np.random.randint(0, 1023, (20, 20, 3), dtype=np.uint16)
        confidence = np.random.rand(20, 20).astype(np.float32)
        out_path = tmp_path / "read_conf.tif"
        write_geotiff(out_path, image, confidence)
        result = read_geotiff_analysis(out_path)
        assert result["bands"] == 4


class TestWriteAnalysisReadyProduct:
    def test_write_with_ars(self, tmp_path):
        from src.evaluation.geotiff_output import write_analysis_ready_product, read_geotiff_analysis
        image = np.random.randint(0, 1023, (20, 20, 3), dtype=np.uint16)
        conf = np.random.rand(20, 20).astype(np.float32)
        ars_result = {
            "ars": 0.85,
            "components": {"confidence": 0.9, "ndvi_preservation": 0.8, "structural_similarity": 0.85},
            "weights": {"confidence": 0.4, "ndvi": 0.3, "structural": 0.3},
        }
        out_path = tmp_path / "ars_product.tif"
        write_analysis_ready_product(out_path, image, conf, ars_result)
        assert out_path.exists()

        result = read_geotiff_analysis(out_path)
        assert "ars" in result
        assert np.isclose(result["ars"], 0.85)
        assert result["ars_components"]["confidence"] == 0.9


class TestMakeProfile:
    def test_default_profile_creation(self):
        from src.evaluation.geotiff_output import _make_geotiff_profile
        import numpy as np
        profile = _make_geotiff_profile(None, 100, 200, 4, np.uint16)
        assert profile["height"] == 100
        assert profile["width"] == 200
        assert profile["count"] == 4
        assert profile["dtype"] == np.uint16
        assert profile["driver"] == "GTiff"

    def test_profile_preserves_existing_keys(self):
        from src.evaluation.geotiff_output import _make_geotiff_profile
        import numpy as np
        existing = {"crs": None, "transform": None, "compress": "lzw"}
        profile = _make_geotiff_profile(existing, 50, 50, 3, np.uint16)
        assert profile["compress"] == "lzw"
        assert profile["crs"] is None
