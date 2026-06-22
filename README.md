# CloudReconstruct

**Adaptive Multi-Source Cloud Removal for LISS-IV Satellite Imagery**

A generative AI framework for the Bharatiya Antariksh Hackathon (BAH) — Problem Statement 2: *Generative AI-Based Cloud Removal and Reconstruction for LISS-IV Satellite Imagery*.

---

## Problem

ISRO's LISS-IV sensor (ResourceSat-2/2A) provides 5.8 m resolution imagery in 3 bands (Green, Red, NIR). Persistent cloud cover over NE India (>70% during monsoon) renders most acquisitions unusable for:
- Land-use / Land-cover mapping
- Agriculture monitoring
- Disaster response
- Environmental assessment
- Infrastructure analysis

## Solution

**CloudReconstruct** uses an adaptive cloud strategy:
1. **Cloud Density Estimator** classifies patches as thin / medium / dense
2. **Adaptive Reconstruction** applies different strategies per density class
3. **Multi-modal Fusion** combines LISS-IV + Sentinel-1 SAR + Sentinel-2 + SRTM DEM
4. **Analysis Readiness Score** provides a single interpretable quality metric

## Project Structure

```
C:\Users\COIN\Desktop\Hackathon
├── data/
│   ├── raw/          # Raw downloaded data (LISS-IV, S1, S2, DEM)
│   ├── processed/    # Aligned, masked, patched data
│   └── outputs/      # Cloud-free results, confidence maps, GeoTIFFs
├── src/
│   ├── preprocessing/  # download, align, cloud_mask, patch_generator
│   ├── models/         # cloud_density, temporal_fusion, sar_diffusion, confidence
│   ├── training/       # train scripts, losses
│   ├── evaluation/     # metrics, ndvi, ars
│   └── app/            # Streamlit web demo
├── notebooks/         # Jupyter notebooks for exploration & testing
├── checkpoints/       # Trained model weights
├── reports/           # Papers, diagrams, architecture, final PPT
├── requirements.txt
├── main.py            # Pipeline entry point
└── README.md
```

## Setup

```bash
cd "C:\Users\COIN\Desktop\Hackathon"
pip install -r requirements.txt
```

## Usage

### Run full data pipeline

```bash
python main.py
```

### Run individual steps

```bash
python main.py --step download   # Check / download data
python main.py --step align      # Co-register scenes
python main.py --step mask       # Generate cloud masks
python main.py --step patch      # Create training patches
```

### Notebooks

```bash
jupyter notebook notebooks/data_exploration.ipynb
```

## Data Sources

| Source | Platform | Resolution | Bands |
|---|---|---|---|
| LISS-IV | Bhoonidhi (ISRO/NRSC) | 5.8 m | Green, Red, NIR |
| Sentinel-1 | Copernicus Data Space | 10 m | VV, VH |
| Sentinel-2 | Copernicus Data Space | 10 m | R, G, B, NIR |
| SRTM DEM | OpenTopography / USGS | 30 m | Elevation |

## Pipeline Stages

1. **Download** — Fetch LISS-IV, Sentinel-1/2, and SRTM data
2. **Align** — Co-register multi-temporal scenes to a common reference
3. **Cloud Mask** — Multi-strategy cloud detection (NDVI, brightness, whiteness, temporal)
4. **Patch** — Sliding window extraction into 256×256 patches with train/val/test split

## Target Metrics

| Metric | Target |
|---|---|
| PSNR (cloud region) | >32 dB |
| SSIM | >0.92 |
| SAM | <3° |
| NDVI correlation | >0.95 |
| Analysis Readiness Score | >90% |

## Team

- 2 × AI & Data Science
- 1 × Electrical Engineering
- 1 × Robotics

## License

Hackathon project — for evaluation purposes only.
