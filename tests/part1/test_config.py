import pytest
from src.config import (
    BASE_DIR, RAW_DATA, PROCESSED, OUTPUTS, CHECKPOINTS,
    LISS4_RAW, S1_RAW, S2_RAW, DEM_RAW,
    ALIGNED, CLOUD_MASKS, PATCHES, MERGED,
    CLOUD_FREE, CONF_MAPS, GEOTIFF_OUT,
    DENSITY_CKPT, DIFFUSION_CKPT, BEST_MODELS,
    LISS4_BANDS, LISS4_RESOLUTION, LISS4_SWATH_MX, LISS4_QUANTIZATION,
    PATCH_SIZE, PATCH_STRIDE, RANDOM_SEED,
)


class TestConfigPaths:
    def test_base_dir_exists(self):
        assert BASE_DIR.exists(), f"BASE_DIR {BASE_DIR} does not exist"

    def test_all_raw_paths_resolve(self):
        for path in [LISS4_RAW, S1_RAW, S2_RAW, DEM_RAW]:
            assert str(path).startswith(str(BASE_DIR))
            assert "raw" in str(path)

    def test_all_processed_paths_resolve(self):
        for path in [ALIGNED, CLOUD_MASKS, PATCHES, MERGED]:
            assert str(path).startswith(str(PROCESSED))

    def test_all_output_paths_resolve(self):
        for path in [CLOUD_FREE, CONF_MAPS, GEOTIFF_OUT]:
            assert str(path).startswith(str(OUTPUTS))

    def test_all_checkpoint_paths_resolve(self):
        for path in [DENSITY_CKPT, DIFFUSION_CKPT, BEST_MODELS]:
            assert str(path).startswith(str(CHECKPOINTS))


class TestLISS4Params:
    def test_band_count(self):
        assert len(LISS4_BANDS) == 3

    def test_band_keys(self):
        for band in ["green", "red", "nir"]:
            assert band in LISS4_BANDS

    def test_band_indices(self):
        assert LISS4_BANDS["green"]["index"] == 0
        assert LISS4_BANDS["red"]["index"] == 1
        assert LISS4_BANDS["nir"]["index"] == 2

    def test_band_wavelengths(self):
        assert LISS4_BANDS["green"]["wavelength"] == (0.52, 0.59)
        assert LISS4_BANDS["red"]["wavelength"] == (0.62, 0.68)
        assert LISS4_BANDS["nir"]["wavelength"] == (0.77, 0.86)

    def test_band_centers(self):
        assert LISS4_BANDS["nir"]["center"] == 0.815

    def test_resolution(self):
        assert LISS4_RESOLUTION == 5.8

    def test_swath(self):
        assert LISS4_SWATH_MX == 23.5

    def test_quantization(self):
        assert LISS4_QUANTIZATION == 10


class TestTrainingDefaults:
    def test_patch_size(self):
        assert PATCH_SIZE == 256
        assert PATCH_STRIDE == 128

    def test_random_seed(self):
        assert RANDOM_SEED == 42
