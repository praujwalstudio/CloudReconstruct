import numpy as np
from pathlib import Path


class TestComputeSlopeAspect:
    def test_output_shapes(self):
        from src.evaluation.dem_integration import compute_slope_aspect
        dem = np.random.rand(50, 50).astype(np.float32)
        slope, aspect = compute_slope_aspect(dem, resolution=5.8)
        assert slope.shape == (50, 50)
        assert aspect.shape == (50, 50)

    def test_slope_non_negative(self):
        from src.evaluation.dem_integration import compute_slope_aspect
        dem = np.random.rand(30, 30).astype(np.float32) * 100
        slope, _ = compute_slope_aspect(dem, resolution=5.8)
        assert slope.min() >= 0

    def test_aspect_range(self):
        from src.evaluation.dem_integration import compute_slope_aspect
        dem = np.random.rand(30, 30).astype(np.float32) * 100
        _, aspect = compute_slope_aspect(dem, resolution=5.8)
        assert aspect.min() >= 0
        assert aspect.max() <= 2 * np.pi

    def test_flat_terrain_zero_slope(self):
        from src.evaluation.dem_integration import compute_slope_aspect
        dem = np.ones((20, 20), dtype=np.float32) * 500
        slope, _ = compute_slope_aspect(dem, resolution=5.8)
        assert np.allclose(slope, 0, atol=1e-6)


class TestCosineCorrection:
    def test_output_shape(self):
        from src.evaluation.dem_integration import cosine_correction
        image = np.random.randint(100, 500, (50, 50, 3), dtype=np.uint16)
        slope = np.random.rand(50, 50) * 0.5
        aspect = np.random.rand(50, 50) * 2 * np.pi
        corrected = cosine_correction(image, slope, aspect, np.radians(45), np.radians(180))
        assert corrected.shape == (50, 50, 3)
        assert corrected.dtype == np.uint16

    def test_no_change_on_flat(self):
        from src.evaluation.dem_integration import cosine_correction
        image = np.ones((20, 20, 3), dtype=np.uint16) * 500
        slope = np.zeros((20, 20))
        aspect = np.zeros((20, 20))
        corrected = cosine_correction(image, slope, aspect, np.radians(45), np.radians(180))
        assert np.allclose(corrected, image, atol=1)

    def test_all_non_negative(self):
        from src.evaluation.dem_integration import cosine_correction
        image = np.random.randint(0, 1000, (30, 30, 3), dtype=np.uint16)
        slope = np.random.rand(30, 30) * np.pi / 4
        aspect = np.random.rand(30, 30) * 2 * np.pi
        corrected = cosine_correction(image, slope, aspect, np.radians(30), np.radians(150))
        assert corrected.min() >= 0


class TestCCorrection:
    def test_output_shape(self):
        from src.evaluation.dem_integration import c_correction
        image = np.random.randint(100, 500, (3, 20, 20), dtype=np.uint16)
        slope = np.random.rand(20, 20) * 0.5
        aspect = np.random.rand(20, 20) * 2 * np.pi
        corrected = c_correction(image, slope, aspect, np.radians(45), np.radians(180))
        assert corrected.shape == (3, 20, 20)

    def test_all_non_negative(self):
        from src.evaluation.dem_integration import c_correction
        image = np.random.randint(0, 1000, (3, 20, 20), dtype=np.uint16)
        slope = np.random.rand(20, 20) * np.pi / 6
        aspect = np.random.rand(20, 20) * 2 * np.pi
        corrected = c_correction(image, slope, aspect, np.radians(30), np.radians(150))
        assert corrected.min() >= 0


class TestDetectTerrainShadows:
    def test_output_shape_and_type(self):
        from src.evaluation.dem_integration import detect_terrain_shadows
        dem = np.random.rand(30, 30).astype(np.float32) * 500
        slope = np.random.rand(30, 30) * 0.5
        aspect = np.random.rand(30, 30) * 2 * np.pi
        shadow = detect_terrain_shadows(dem, slope, aspect, np.radians(45), np.radians(180))
        assert shadow.shape == (30, 30)
        assert shadow.dtype == np.uint8
        assert set(np.unique(shadow)).issubset({0, 1})

    def test_no_shadows_at_zenith(self):
        from src.evaluation.dem_integration import detect_terrain_shadows
        dem = np.ones((20, 20), dtype=np.float32) * 100
        slope = np.zeros((20, 20))
        aspect = np.zeros((20, 20))
        shadow = detect_terrain_shadows(dem, slope, aspect, np.radians(90), np.radians(0))
        assert shadow.sum() >= 0


class TestTerrainProcessor:
    def test_requires_load(self):
        from src.evaluation.dem_integration import TerrainProcessor
        tp = TerrainProcessor()
        import pytest
        with pytest.raises(ValueError, match="Call load"):
            tp.correct(np.ones((10, 10, 3), dtype=np.uint16))
        with pytest.raises(ValueError, match="Call load"):
            tp.shadow_mask()

    def test_load_from_array(self, tmp_path):
        from src.evaluation.dem_integration import TerrainProcessor, load_dem

        dem = np.ones((20, 20), dtype=np.float32) * 100
        dem_path = tmp_path / "test_dem.tif"
        import rasterio
        from rasterio.transform import from_origin
        profile = {
            "driver": "GTiff", "height": 20, "width": 20,
            "count": 1, "dtype": np.float32,
            "crs": None, "transform": from_origin(0, 0, 1, 1),
        }
        with rasterio.open(dem_path, "w", **profile) as dst:
            dst.write(dem, 1)

        tp = TerrainProcessor(resolution=1.0)
        tp.load(dem_path)
        assert tp.dem is not None
        assert tp.slope is not None
        assert tp.aspect is not None
        assert tp.slope.shape == (20, 20)

    def test_cosine_correction_through_processor(self, tmp_path):
        from src.evaluation.dem_integration import TerrainProcessor

        dem_path = tmp_path / "dem.tif"
        import rasterio
        from rasterio.transform import from_origin
        with rasterio.open(dem_path, "w", driver="GTiff", height=20, width=20,
                           count=1, dtype=np.float32, crs=None,
                           transform=from_origin(0, 0, 1, 1)) as dst:
            dst.write(np.ones((20, 20), dtype=np.float32) * 100, 1)

        tp = TerrainProcessor(resolution=1.0)
        tp.load(dem_path)
        image = np.random.randint(100, 500, (20, 20, 3), dtype=np.uint16)
        corrected = tp.correct(image, method="cosine")
        assert corrected.shape == (20, 20, 3)

    def test_shadow_mask_through_processor(self, tmp_path):
        from src.evaluation.dem_integration import TerrainProcessor

        dem_path = tmp_path / "dem.tif"
        import rasterio
        from rasterio.transform import from_origin
        with rasterio.open(dem_path, "w", driver="GTiff", height=20, width=20,
                           count=1, dtype=np.float32, crs=None,
                           transform=from_origin(0, 0, 1, 1)) as dst:
            dst.write(np.random.rand(20, 20).astype(np.float32) * 200, 1)

        tp = TerrainProcessor(resolution=1.0)
        tp.load(dem_path)
        shadow = tp.shadow_mask()
        assert shadow.shape == (20, 20)
        assert shadow.dtype == np.uint8
