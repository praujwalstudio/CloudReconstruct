import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pytest


@pytest.fixture
def sample_liss4_image() -> np.ndarray:
    h, w = 256, 256
    image = np.zeros((h, w, 3), dtype=np.uint16)

    # Green band (some vegetation-like pattern)
    image[..., 0] = np.random.randint(200, 500, (h, w)).astype(np.uint16)
    # Red band
    image[..., 1] = np.random.randint(100, 300, (h, w)).astype(np.uint16)
    # NIR band (bright — healthy vegetation)
    image[..., 2] = np.random.randint(500, 800, (h, w)).astype(np.uint16)

    return image


@pytest.fixture
def sample_cloudy_image() -> np.ndarray:
    h, w = 256, 256
    image = np.ones((h, w, 3), dtype=np.uint16) * 800
    # Add a dark rectangle as ground feature
    image[50:200, 50:200, :] = 200
    return image


@pytest.fixture
def sample_clear_image() -> np.ndarray:
    h, w = 256, 256
    image = np.ones((h, w, 3), dtype=np.uint16) * 200
    # Add a brighter rectangle as feature
    image[50:200, 50:200, :] = 400
    return image


@pytest.fixture
def sample_binary_mask() -> np.ndarray:
    mask = np.zeros((256, 256), dtype=np.uint8)
    mask[50:150, 50:150] = 1
    return mask


@pytest.fixture
def temp_output_dir(tmp_path) -> Path:
    return tmp_path / "test_outputs"
