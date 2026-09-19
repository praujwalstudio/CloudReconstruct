import os
from pathlib import Path

# Base directory: default to repository root or environment variable
BASE_DIR = Path(os.environ.get("CLOUDRECONSTRUCT_BASE_DIR", Path(__file__).resolve().parent.parent))

# Raw data
RAW_DATA = BASE_DIR / "data" / "raw"
RAW = RAW_DATA
LISS4_RAW = RAW_DATA / "liss4"
S1_RAW = RAW_DATA / "sentinel1"
S2_RAW = RAW_DATA / "sentinel2"
DEM_RAW = RAW_DATA / "dem"
# SEN12MS_RAW points to the local multi-modal data (cloudy/clear/sigma0/dem folders)
SEN12MS_RAW = RAW_DATA
SEN12MS_COMPACT = RAW_DATA / "sen12ms_cr" / "compact"

# Processed data
PROCESSED = BASE_DIR / "data" / "processed"
ALIGNED = PROCESSED / "aligned"
CLOUD_MASKS = PROCESSED / "cloud_masks"
PATCHES = PROCESSED / "patches"
MERGED = PROCESSED / "merged"

# Outputs
OUTPUTS = BASE_DIR / "data" / "outputs"
CLOUD_FREE = OUTPUTS / "cloud_free"
CONF_MAPS = OUTPUTS / "confidence_maps"
GEOTIFF_OUT = OUTPUTS / "geotiff"

# Checkpoints
CHECKPOINTS = BASE_DIR / "checkpoints"
DENSITY_CKPT = CHECKPOINTS / "density_model"
DIFFUSION_CKPT = CHECKPOINTS / "diffusion_model"
BEST_MODELS = CHECKPOINTS / "best_models"

# Reports
REPORTS = BASE_DIR / "reports"
PAPERS = REPORTS / "papers"
DIAGRAMS = REPORTS / "diagrams"
ARCHITECTURE = REPORTS / "architecture"
FINAL_PPT = REPORTS / "final_ppt"

# LISS-IV sensor parameters
LISS4_BANDS = {
    "green": {"index": 0, "wavelength": (0.52, 0.59), "center": 0.555},
    "red": {"index": 1, "wavelength": (0.62, 0.68), "center": 0.650},
    "nir": {"index": 2, "wavelength": (0.77, 0.86), "center": 0.815},
}
LISS4_RESOLUTION = 5.8  # meters
LISS4_SWATH_MX = 23.5   # km
LISS4_QUANTIZATION = 10  # bits

# Training defaults
RANDOM_SEED = 42
PATCH_SIZE = 256
PATCH_STRIDE = 128
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1
