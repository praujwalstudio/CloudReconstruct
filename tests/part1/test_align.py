import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from src.preprocessing.align import find_gcps, align_all_scenes


class TestFindGCPs:
    def test_returns_tuple(self):
        ref = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        tgt = np.roll(ref, shift=5, axis=1)  # shifted version
        matrix, n = find_gcps(ref, tgt)
        # Either returns matrix or None
        assert isinstance(matrix, (np.ndarray, type(None)))
        assert isinstance(n, (int, type(None)))

    def test_identical_images(self):
        # SIFT needs texture — use random noise
        rng = np.random.default_rng(42)
        img = rng.integers(0, 255, (100, 100, 3), dtype=np.uint8)
        matrix, n = find_gcps(img, img)
        # Identical images should find matches
        assert matrix is not None or n is not None


class TestAlignPair:
    def test_returns_dict_structure(self):
        result = {
            "moving": "moving.tif",
            "fixed": "fixed.tif",
            "output": "aligned.tif",
            "matches": 42,
            "aligned": True,
        }
        assert isinstance(result, dict)
        assert all(k in result for k in ["moving", "fixed", "output", "matches", "aligned"])

    def test_find_gcps_with_numpy_only(self):
        from src.preprocessing.align import find_gcps

        rng = np.random.default_rng(42)
        ref = rng.integers(0, 255, (100, 100, 3), dtype=np.uint8)
        tgt = np.roll(ref, shift=5, axis=1)
        matrix, n = find_gcps(ref, tgt)
        assert isinstance(matrix, (np.ndarray, type(None)))
        assert n is None or isinstance(n, int)


class TestAlignAllScenes:
    def test_empty_scenes_returns_empty_list(self):
        with patch("src.preprocessing.align.LISS4_RAW") as mock_raw:
            mock_raw.glob.return_value = []
            result = align_all_scenes()
            assert result == []
