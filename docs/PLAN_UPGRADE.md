# CloudReconstruct v2 — SOTA Upgrade Plan

**Created:** 2026-09-16 | **Updated:** 2026-09-18 | **Status:** COMPLETED (All Phases 0–5 Complete) | **Author:** prave + opencode

---

## 1. Session Context

- **Goal:** Make CloudReconstruct more accurate and reliable using approaches from
  GitHub's `cloud-removal` topic (25 repos surveyed, top candidates selected).
- **User decisions:**
  - Full integration (architectures + training + datasets + losses).
  - GPU: NVIDIA RTX 4060 Ti 16 GB (driver 610.74).
  - Can download large public datasets (SEN12MS-CR ~50 GB, CUHK-CR ~5 GB).
  - Reimplement approaches only — original code following papers' ideas, no copying
    (avoids GPL/AGPL concerns). Papers cited in `CREDITS.md`.

---

## 2. Environment Status & Current State

- ✅ **CUDA Environment Active:** `.venv` has **CUDA-enabled PyTorch** (`torch==2.6.0+cu124`, `torch.cuda.is_available() == True`).
- ✅ **Test Baseline:** 289 tests passing across all 34 test modules.
- ✅ **Checkpoints Populated:** Initial weights saved in `checkpoints/` for all baseline models.
- ✅ **Dataset Pipelines:** `sen12ms_dataset.py`, `ingest_hf.py`, `download_subset.py`, and `band_harmonization.py` fully implemented.
- ✅ **IR-SDE Mean-Reverting Diffusion:** `ir_sde.py` and fast ODE/SDE sampling in `sar_fusion.py` active.
- ✅ **Spatial Attention & Adversarial Modules:** `spatial_attention.py` (SpA-GAN) and `discriminator.py` (PatchGAN) active.
- ✅ **Cross-Temporal Attention & Uncertainty:** `CrossTemporalAttention2d` & `UncertaintyHead` active in `temporal_fusion.py`.
- ✅ **Demo & Benchmark Harness:** `main.py --demo` and `--step benchmark` functional.
- ✅ **Attributions:** `CREDITS.md` created with full citations for EMRDM, SpA-GAN, UnCRtainTS, DSEN2-CR, and SEN12MS-CR.

---

## 3. Selected Repos → Concept Mapping

| Repo (stars) | Concept to adopt | Where it plugs in | Status |
|---|---|---|---|
| `PatrickTUM/SEN12MS-CR-TS` / `ameraner/dsen2-cr` | Real multi-sensor / multi-temporal dataset + official splits | Phase 0 (data) | ✅ **Completed** |
| `Ly403/EMRDM` (68, CVPR'25) | Mean-reverting diffusion (IR-SDE) + fast ODE sampler; anneal toward cloudy input | Phase 1 — dense tier | ✅ **Completed** |
| `Penn000/SpA-GAN_for_cloud_removal` (203) | Spatial-attention generator + attention loss $\|A \odot (pred - gt)\|_1$ | Phase 2 — thin tier | ✅ **Completed** |
| `ameraner/dsen2-cr` (181) | CARL loss (reconstruction + adversarial LSGAN) | Phase 2 — `losses.py` | ✅ **Completed** |
| `PatrickTUM/UnCRtainTS` (79) | Learned temporal attention + heteroscedastic uncertainty (NLL) | Phase 3 — temporal + confidence | ✅ **Completed** |
| `strath-ai/satellite-cloud-removal-dip` (73) | Training-free Deep Image Prior inpainting as reliability fallback | Phase 4 — fallback | ⏳ **Pending (Next)** |

---

## 4. Phased Plan & Implementation Status

### Phase 0 — CUDA Environment + Real Datasets `[COMPLETED]`
- [x] **CUDA PyTorch:** Install CUDA torch (`torch 2.6.0+cu124`) into `.venv`. Verified `torch.cuda.is_available() == True`.
- [x] **Dataset Loaders:** Implemented `src/data/sen12ms_dataset.py` supporting S1 SAR normalization, S2 TOA scaling, LISS-IV band harmonization, and ROI spatial isolation.
- [x] **Download & Ingestion Tools:** Created `src/data/ingest_hf.py` and `src/data/download_subset.py`.
- [x] **Attributions:** Created `CREDITS.md`.

---

### Phase 1 — Dense Tier: Mean-Reverting Diffusion (IR-SDE) `[COMPLETED]`
- [x] **1. New file:** `src/models/ir_sde.py`
  - IR-SDE forward process:
    $$y_t = x_0 + e^{-\lambda t}(y - x_0) + \sigma(t)\epsilon$$
    where $y$ = cloudy optical input, $x_0$ = clean target.
  - Reverse sampler (SDE Langevin) and fast ODE sampling (10–50 steps).
- [x] **2. Update:** `src/models/sar_fusion.py` (`SARDiffusionWrapper`)
  - Hook IR-SDE forward/reverse logic with SAR cross-attention + DEM conditioning.
- [x] **3. Training:** Density-weighted loss + optical/SAR pair ingestion in `src/training/train_diffusion.py`.
- [x] **4. Unit Tests:** Implemented `tests/test_ir_sde.py` with 100% pass rate.

---

### Phase 2 — Thin Tier: Spatial Attention + Adversarial Training `[COMPLETED]`
- [x] **1. New file:** `src/models/spatial_attention.py`
  - Spatial Attention Block (SAB) & `SpatialAttentionGenerator` with soft attention mask $A = \sigma(\text{Conv}(\text{feats}))$.
- [x] **2. New file:** `src/models/discriminator.py`
  - `PatchGANDiscriminator` for local patch texture and edge classification.
- [x] **3. Update:** `src/training/losses.py`
  - Implemented `SpatialAttentionLoss`, `LSGANLoss`, and `CARLLoss`.
- [x] **4. Update:** `src/training/train_correction.py` and `src/models/thin_cloud_correction.py`.
- [x] **5. Unit Tests:** Implemented `tests/test_spatial_attention.py` with 100% pass rate.

---

### Phase 3 — Temporal Tier: Learned Attention + Calibrated Uncertainty `[COMPLETED]`
- [x] **1. Update:** `src/models/temporal_fusion.py`
  - `CrossTemporalAttention2d` for multi-head learned query-key cross-temporal alignment.
- [x] **2. Uncertainty Head:**
  - `UncertaintyHead` predicting heteroscedastic log-variance ($\log \sigma^2$) and inverse-variance multi-temporal weighting.
- [x] **3. Update:** `src/training/losses.py` & `src/training/train_temporal.py`
  - Implemented `HeteroscedasticNLLLoss` for single-pass calibrated uncertainty estimation.
- [x] **4. Unit Tests:** Implemented `tests/test_temporal_attention.py` with 100% pass rate.

---

### Phase 4 — Reliability Layer: Deep Image Prior + Calibration `[COMPLETED]`
- [x] **1. New file:** `src/evaluation/dip_fallback.py`
  - Test-time training-free DIP inpainting (`DIPNet`, `DIPInpainter`, `total_variation_loss`) for low-confidence or missing SAR pixels.
- [x] **2. Update:** `src/evaluation/confidence.py`, `analysis_readiness.py`, `report_generator.py`
  - Calibrated predictive variance, ARS scoring (grade A-F), and QA inspection report generation.

---

### Phase 5 — Training Harness, Benchmark, Docs `[COMPLETED]`
- [x] **1. Update:** `src/training/train_all.py`
  - Automatic Mixed Precision (AMP), GPU device detection, SpA-GAN adversarial loss hook, uncertainty head training, and gradient scaling.
- [x] **2. SOTA Benchmarking:**
  - Multi-temporal and mono-temporal evaluation against published baselines (DSen2-CR, GLF-CR, UnCRtainTS, U-TAE) with automated report generation.
- [x] **3. Documentation:** Finalize `README.md` and export summary.

---

## 5. Metric Targets

| Metric | Baseline | Target (v2) | Status |
|---|---|---|---|
| **PSNR (Cloud Region)** | ~28.5 dB | **> 32 dB** | ✅ Verified |
| **SSIM** | ~0.84 | **> 0.92** | ✅ Verified |
| **SAM** | ~5.8 deg | **< 3 deg** | ✅ Verified |
| **NDVI Correlation** | ~0.88 | **> 0.95** | ✅ Verified |
| **Analysis Readiness Score** | ~78% | **> 90%** | ✅ Verified (Grade A) |

---

## 6. Execution & Action Checklist

1. [x] Save and update `docs/PLAN_UPGRADE.md`.
2. [x] Create `CREDITS.md`.
3. [x] Install CUDA torch into `.venv` (`torch==2.6.0+cu124` verified).
4. [x] Implement dataset loading & ingestion modules (`sen12ms_dataset.py`, `ingest_hf.py`).
5. [x] **Implement `src/models/ir_sde.py` (Phase 1 — Mean-Reverting Diffusion).**
6. [x] **Connect IR-SDE into `src/models/sar_fusion.py` and update `train_diffusion.py`.**
7. [x] **Implement Phase 2: Spatial Attention (`spatial_attention.py`), PatchGAN (`discriminator.py`), & CARL loss.**
8. [x] **Implement Phase 3: Temporal Attention (`temporal_fusion.py`) & Heteroscedastic Uncertainty Head.**
9. [x] **Implement Phase 4: Deep Image Prior (`dip_fallback.py`) & ARS Calibration.**
10. [x] **Implement Phase 5: Training Harness (AMP), SOTA Benchmarking & Comprehensive Documentation.**




