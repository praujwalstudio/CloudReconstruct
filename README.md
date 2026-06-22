# ☁️ CloudReconstruct

**Adaptive Multi-Source Cloud Removal for LISS-IV Satellite Imagery**

*Bharatiya Antariksh Hackathon — Problem Statement 2: Generative AI-Based Cloud Removal and Reconstruction for LISS-IV Satellite Imagery*

---

📰 News
---
✅ Full project implementation released — core models, training pipeline, evaluation, and web demo.

✅ Unified training CLI (`src/training/train_all.py`) — train all 4 models with a single command.

✅ Streamlit web demo (`src/app/app.py`) — interactive upload, correct, and download.

✅ CI pipeline (GitHub Actions) + Docker support for reproducible deployment.

---

🎯 Overview
---
CloudReconstruct is an adaptive, multi-source framework for removing clouds from ISRO's LISS-IV imagery (5.8 m, 3‑band: Green, Red, NIR). It estimates cloud density per pixel, then applies a tiered reconstruction strategy:

| Density | Strategy | Inputs |
|---------|----------|--------|
| Thin    | Residual correction | LISS-IV only |
| Medium  | Temporal fusion | LISS-IV + temporal reference(s) |
| Dense   | SAR-guided diffusion | LISS-IV + Sentinel‑1 SAR |

A 5‑class cloud mask (clear / thin / medium / thick / shadow) and an uncertainty‑aware confidence map accompany every output.

---

🔧 Setup
---

```bash
git clone https://github.com/praujwalstudio/CloudReconstruct.git
cd CloudReconstruct
pip install -r requirements.txt
```

> **Note**: PyTorch, rasterio, and scikit-image are the key dependencies. CUDA is auto‑detected; falls back to CPU.

---

📌 Data Sources
---

| Source | Platform | Bands | Resolution |
|--------|----------|-------|------------|
| LISS‑IV | Bhoonidhi (ISRO/NRSC) | Green, Red, NIR | 5.8 m |
| Sentinel‑1 | Copernicus Data Space | VV, VH | 10 m |
| Sentinel‑2 | Copernicus Data Space | R, G, B, NIR | 10 m |
| SRTM DEM | OpenTopography / USGS | Elevation | 30 m |

---

🔎 Architecture
---

```
LISS‑IV ──► CloudDensityNet ──► density map
                │
    ┌───────────┼───────────┐
    │ thin      │ medium    │ dense
    ▼           ▼           ▼
ThinCloudCorrection    TemporalFusion    SARConditionalUNet
(residual)       (self-attn blend)  (cross-attn + DDPM)
    │           │           │
    └───────────┼───────────┘
                ▼
         AdaptiveBlend
                ▼
         Cloud‑free output
```

- **CloudDensityNet** — U‑Net with Monte‑Carlo Dropout; outputs density + predictive uncertainty.
- **ThinCloudCorrection** — learns a residual correction modulated by density.
- **TemporalFusion** — aligns and blends a temporal reference via optical flow + self‑attention.
- **SARConditionalUNet** — cross‑attention between LISS‑IV and SAR features; optionally runs as a DDPM.
- **Confidence** — combines density, temporal variance, SAR coherence, terrain shadow, and MC‑dropout uncertainty.

---

🔥 Training
---

All four models can be trained with the unified CLI:

```bash
# Train all models sequentially (density → correction → temporal → diffusion)
python -m src.training.train_all --epochs 50 --batch-size 8

# Train only the density estimator
python -m src.training.train_all --model density --epochs 20

# Train correction with a pre‑trained density checkpoint
python -m src.training.train_all --model correction --epochs 30 --with-density

# Train the SAR diffusion model
python -m src.training.train_all --model diffusion --epochs 100 --noise-steps 100

# Use GPU
python -m src.training.train_all --device cuda --epochs 100
```

Each model can also be trained individually via its own `__main__` block:

```bash
python -m src.training.train_density
python -m src.training.train_correction
python -m src.training.train_temporal
python -m src.training.train_diffusion
```

Training logs and checkpoints are saved under `checkpoints/`.

---

🏃 Inference
---

### CLI pipeline

```bash
# Full pipeline (download → align → mask → patch → infer)
python main.py

# Preview steps without executing
python main.py --dry-run

# Run a single step
python main.py --step mask
python main.py --step infer
```

### Programmatic usage

```python
from src.evaluation.inference import CloudFreeInference

model = CloudFreeInference(device="cpu")
result = model.correct(
    image=liss4_array,          # uint16 (H, W, 3)
    sar=sar_array,              # optional, (H, W, 2)
    temporal_refs=[ref_array],  # optional
    dem_processor=processor,    # optional TerrainProcessor
    data_max=1023.0,            # radiometric max for LISS‑4
)
# result = {"corrected", "density", "confidence", "ars", "dtype"}
```

---

💻 Web Demo
---

```bash
streamlit run src/app/app.py
```

Upload a LISS‑IV GeoTIFF, optionally add SAR / DEM, and download the cloud‑free result with interactive visualisations of density, confidence, and metrics.

---

📊 Target Metrics
---

| Metric | Target |
|--------|--------|
| PSNR (cloud region) | >32 dB |
| SSIM | >0.92 |
| SAM | <3° |
| NDVI correlation | >0.95 |
| Analysis Readiness Score | >90 % |

---

📂 Project Structure
---

```
CloudReconstruct/
├── src/
│   ├── preprocessing/    # download, align, cloud_mask, patch_generator
│   ├── models/           # cloud_density, thin_cloud_correction,
│   │                     # temporal_fusion, sar_fusion, adaptive_pipeline
│   ├── training/         # losses, synthetic_data, train_*.py
│   ├── evaluation/       # inference, metrics, confidence, analysis_readiness,
│   │                     # dem_integration, geotiff_output
│   └── app/              # Streamlit web application
├── tests/                # 209 unit & integration tests
│   ├── part1/ … part5/
│   └── e2e/
├── main.py               # CLI pipeline entry point
├── requirements.txt
├── Dockerfile
└── .github/workflows/    # CI (Ubuntu + Windows, Python 3.11)
```

---

✨ Acknowledgment
---
This project uses LISS‑IV data from ISRO's Bhoonidhi portal, Sentinel‑1/2 data from the Copernicus programme, and SRTM DEM from NASA/USGS. Built with PyTorch, Streamlit, rasterio, and scikit‑image.

---

📖 BibTeX
---

```bibtex
@misc{cloudreconstruct2025,
  title={CloudReconstruct: Adaptive Multi-Source Cloud Removal for LISS-IV Satellite Imagery},
  author={CloudReconstruct Team},
  year={2025},
  note={Bharatiya Antariksh Hackathon — Problem Statement 2},
}
```
