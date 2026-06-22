# CloudReconstruct: Multi-Source Cloud Removal for Satellite Imagery

An end-to-end deep learning pipeline for adaptive cloud removal, thin cloud correction, temporal fusion alignment, and SAR-conditional diffusion refinement. This repository contains the complete source code required to build the environment, structure the data, and train all four sequential models from scratch.

---

## System Prerequisites & Installation

This project requires Python 3.10+ and a CUDA-capable GPU is highly recommended for training.

```bash
# 1. Clone the repository
git clone https://github.com
cd CloudReconstruct

# 2. Initialize an isolated virtual environment
python -m venv .venv

# 3. Activate the environment
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On macOS / Linux:
source .venv/bin/activate

# 4. Install all runtime dependencies
pip install -r requirements.txt
```

---

## Required Project Architecture

Because large dataset files and trained model weights (`.pth`) are ignored by version control to keep the repository lightweight, you must manually set up the following directory structure in your root folder before launching any training scripts:

```text
CloudReconstruct/
├── checkpoints/
│   ├── correction_model/
│   ├── density_model/
│   ├── diffusion_model/
│   └── temporal_model/
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── evaluation/
│   ├── models/
│   └── training/
└── README.md
```

Place your raw satellite scenes (e.g., LISS-IV, Sentinel-1 SAR, or multi-temporal patches) into the `data/raw/` directory.

---

## Step-by-Step Training Pipeline

The execution architecture relies on a strict sequential dependency. Each model relies on inputs or processing rules established by the previous step. Run the training scripts in this exact order:

### Step 1: Cloud Density Estimation (`CloudDensityNet`)
Trains a robust UNet architecture to segment and generate continuous cloud density probability masks from optical bands.
```bash
python src/training/train_density.py
```
**Output:** Saves `best_model.pth` and `final_model.pth` into `checkpoints/density_model/` (~13 MB).

### Step 2: Thin Cloud Correction (`ThinCloudCorrection`)
Trains a fast, specialized 2-layer convolutional network to suppress thin cloud haze while preserving underlying land surface reflectances.
```bash
python src/training/train_correction.py
```
**Output:** Saves `best_model.pth` and `final_model.pth` into `checkpoints/correction_model/` (~48 KB).

### Step 3: Multi-Temporal Fusion Alignment (`TemporalFusion`)
Trains an integrated `AlignmentNet` and `SelfAttention2d` network to dynamically align historical multi-temporal reference imagery and fuse cloud-free spatial pixels.
```bash
python src/training/train_temporal.py
```
**Output:** Saves `best_model.pth` into `checkpoints/temporal_model/` (~286 KB).

### Step 4: SAR Conditional Diffusion (`SARDiffusionWrapper`)
Trains the final generative structural model. It uses conditional SAR (Synthetic Aperture Radar) data inputs to reconstruct highly clouded areas with realistic textures.
```bash
python src/training/train_diffusion.py
```
**Output:** Saves checkpoints up to `model_epoch_10.pth` into `checkpoints/diffusion_model/` (~13 MB).

---

## Evaluation & Inference

Once all checkpoints are populated in the `checkpoints/` folder, you can execute patch-based tiled inference on large optical scenes. The script automatically handles single-channel replication, guards NDVI metrics, and uses overlapping windows to completely eliminate tile-edge seam artifacts.

```bash
python src/evaluation/inference.py --input data/raw/cloudy_scene.tif --output data/processed/reconstructed_scene.tif
```
