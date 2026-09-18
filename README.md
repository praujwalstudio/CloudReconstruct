# ☁️ CloudReconstruct v2 (SOTA)

**Adaptive Multi-Source Cloud Removal & Reconstruction for LISS-IV Satellite Imagery**

*Bharatiya Antariksh Hackathon — Problem Statement 2: Generative AI-Based Cloud Removal and Reconstruction for LISS-IV Satellite Imagery*

---

📰 News & SOTA Upgrades (v2)
---
✅ **Mean-Reverting Diffusion (IR-SDE):** Continuous Mean-Reverting Stochastic Differential Equation inpainting with fast Heun/Euler ODE sampling (10–20 steps) for dense cloud penetration.

✅ **Spatial Attention GAN (SpA-GAN):** Spatial Attention Blocks (SAB) + PatchGAN discriminator + Cloud-Aware Adversarial Representation Learning (CARL) for thin cloud & haze removal.

✅ **Cross-Temporal Attention & Calibrated Uncertainty:** Attention-guided temporal fusion with Heteroscedastic Gaussian NLL loss, predicting per-pixel mean reflectance and aleatoric variance.

✅ **Deep Image Prior (DIP) Zero-Shot Fallback:** Test-time self-supervised optimization layer with Total Variation regularization for low-confidence or missing SAR regions.

✅ **Unified AMP Training & Benchmarking CLI:** Native `torch.amp` Automatic Mixed Precision and automated benchmarking against published baselines (DSen2-CR, GLF-CR, UnCRtainTS, U-TAE).

✅ **Full Test Suite:** 289/289 automated unit, integration, and end-to-end tests passing with 100% green coverage.

---

🎯 Overview
---
CloudReconstruct is an adaptive, multi-source framework for removing clouds from ISRO's LISS-IV imagery (5.8 m, 3‑band: Green, Red, NIR). It estimates cloud density per pixel, then applies a tiered reconstruction strategy:

| Density | Strategy | Architecture | Inputs |
|---|---|---|---|
| **Thin** | Spatial Attention + GAN | `SpatialAttentionNet` (SAB + PatchGAN) | LISS-IV + Density |
| **Medium** | Cross-Temporal Attention + Uncertainty | `TemporalFusion` (Cross-Attention + Uncertainty Head) | LISS-IV + Historical Reference + Density |
| **Dense** | Mean-Reverting Diffusion + SAR Fusion | `MeanRevertingSDE` (IR-SDE + SARConditionalUNet) | LISS-IV + Sentinel-1 SAR + DEM |
| **Reliability** | Deep Image Prior (DIP) Fallback | `DIPInpainter` (Self-Supervised DIP + TV) | Zero-Shot Fallback on High Variance |

A 5‑class cloud mask (clear / thin / medium / thick / shadow), an uncertainty‑aware calibrated confidence map, and an **Analysis-Readiness Score (ARS)** accompany every output GeoTIFF.

---

🔧 Setup & Installation
---

### Prerequisites
- Python 3.10+
- NVIDIA GPU with CUDA recommended for training (e.g., RTX 4060 Ti / A100 / T4)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/praujwalstudio/CloudReconstruct.git
cd CloudReconstruct

# 2. Create and activate a virtual environment
python -m venv .venv
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

> **Note**: PyTorch 2.6+ with CUDA (`torch==2.6.0+cu124`), rasterio, and scikit-image are installed. CUDA and AMP are auto-detected and enabled automatically on GPU environments.

---

📌 Data Sources
---

| Source | Platform | Bands | Resolution |
|---|---|---|---|
| **LISS‑IV** | Bhoonidhi (ISRO/NRSC) | Green, Red, NIR | 5.8 m |
| **Sentinel‑1** | Copernicus Data Space | VV, VH | 10 m |
| **Sentinel‑2 / SEN12MS-CR** | Copernicus Data Space / HF | R, G, B, NIR (Harmonized) | 10 m |
| **SRTM DEM** | OpenTopography / USGS | Elevation, Slope, Aspect, Hillshade | 30 m |

---

🔎 Architecture Pipeline
---

```
LISS‑IV (+ DEM) ──► CloudDensityNet ──► continuous density map + uncertainty
                          │
         ┌────────────────┼────────────────┐
         │ thin           │ medium         │ dense
         ▼                ▼                ▼
SpatialAttentionNet    TemporalFusion    MeanRevertingSDE
 (SAB Residual + GAN) (Cross-Attention) (SAR IR-SDE + DEM)
         │                │                │
         └────────────────┼────────────────┘
                          ▼
                   AdaptiveBlend
                          │
          [If Confidence < Threshold] ──► DIP Zero-Shot Inpainting Fallback
                          ▼
            Analysis-Ready GeoTIFF Product
            (Cloud-Free + Confidence + ARS Grade A-F)
```

---

🔥 Training & Unified CLI
---

Train models individually or sequentially with Automatic Mixed Precision (AMP):

```bash
# Train all 4 models sequentially with AMP
python -m src.training.train_all --epochs 50 --batch-size 8 --use-amp

# Train only CloudDensityNet (with DEM Early Fusion & Filtered Jaccard Loss)
python -m src.training.train_all --model density --epochs 20 --use-fjl

# Train Thin Cloud Correction (with Spatial Attention & PatchGAN adversarial loss)
python -m src.training.train_all --model correction --epochs 30 --use-adv --with-density

# Train Temporal Fusion (with Cross-Attention & Heteroscedastic Uncertainty Head)
python -m src.training.train_all --model temporal --epochs 30 --use-uncertainty

# Train SAR Diffusion (with Mean-Reverting IR-SDE)
python -m src.training.train_all --model diffusion --epochs 50 --noise-steps 100
```

---

📊 SOTA Accuracy Benchmarking
---

Evaluate checkpoints against published benchmarks (Patrick Ebel's SOTA benchmark repository, DSen2-CR, GLF-CR, UnCRtainTS):

```bash
# Run multi-temporal and mono-temporal benchmark suite
python main.py --step benchmark
# or directly:
python -m src.evaluation.benchmark_accuracy --device cpu --output-dir data/outputs
```

Outputs:
- `data/outputs/benchmark_report.md` (Publication-ready markdown table)
- `data/outputs/benchmark_summary.json` (Machine-readable metrics: RMSE, MAE, PSNR, SSIM, SAM)

---

🏃 Automated Demo & CLI Inference
---

```bash
# Run automated reference demonstration on thin, medium, and dense scenes
python main.py --demo

# Run full preprocessing and inference pipeline
python main.py

# Run Streamlit interactive web application
streamlit run src/app/app.py
```

---

🧪 Test Suite
---

Run the comprehensive test suite (289 tests):

```bash
pytest
```

---

✨ Credits & Citations
---
See [`CREDITS.md`](CREDITS.md) for full academic citations of core foundational research (CVPR'25 EMRDM, SpA-GAN, UnCRtainTS, Deep Image Prior, and SEN12MS-CR-TS benchmark baselines).

```bibtex
@misc{cloudreconstruct2025,
  title={CloudReconstruct v2: Adaptive Multi-Source SOTA Cloud Removal for LISS-IV Satellite Imagery},
  author={CloudReconstruct Team},
  year={2025},
  note={Bharatiya Antariksh Hackathon — Problem Statement 2},
}
```
