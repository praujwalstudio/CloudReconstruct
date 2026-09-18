# CREDITS.md — CloudReconstruct

This document lists all published approaches whose concepts are reimplemented
(original code, no direct copying) in CloudReconstruct, along with their
original repos and papers.

---

## Dense cloud tier: IR-SDE / mean-reverting diffusion

**Adopted from:** EMRDM — *Effective Cloud Removal for Remote Sensing Images
by an Improved Mean-Reverting Denoising Model with Elucidated Design Space*
- Paper: Liu, Li, Guan, Zhou, Zhang — CVPR 2025
  https://arxiv.org/abs/2503.23717
- Repo: https://github.com/Ly403/EMRDM  (AGPL-3.0)
- Concept adopted: mean-reverting (IR-SDE) forward process and fast ODE
  sampling for optical+SAR cloud removal. Implemented as original code
  in `src/models/ir_sde.py`.

---

## CARL / adversarial training loss

**Adopted from:** DSen2-CR — *Cloud Removal in Sentinel-2 Imagery Using a
Deep Residual Neural Network and SAR-optical Data Fusion*
- Paper: Meraner, Ebel, Zhu, Schmitt — ISPRS J. Photogrammetry and
  Remote Sensing, 2020  (ISPRS Best Paper / Helava Award)
- Repo: https://github.com/ameraner/dsen2-cr  (GPL-3.0)
- Concept adopted: CARL (Combined Adversarial + Reconstruction Loss)
  and SAR-optical fusion pattern. Implemented as original code in
  `src/training/losses.py` and `src/models/discriminator.py`.

---

## Spatial attention generator + attention loss

**Adopted from:** SpA-GAN — *Cloud Removal for Remote Sensing Imagery via
Spatial Attention Generative Adversarial Network*
- Paper: Pan — arXiv 2020 (arXiv:2009.13015)
- Repo: https://github.com/Penn000/SpA-GAN_for_cloud_removal  (MIT)
- Concept adopted: soft spatial attention map to focus cloud corrections;
  attention loss `|A⊙(pred−gt)|_1`. Implemented as original code in
  `src/models/spatial_attention.py`.

---

## Learned temporal attention + calibrated uncertainty

**Adopted from:** UnCRtainTS — *Uncertainty-Aware Transformer for
Satellite Time Series Cloud Removal*
- Paper: Ebel et al. — CVPR 2023
- Repo: https://github.com/PatrickTUM/UnCRtainTS  (MIT)
- Concept adopted: learned cross-temporal attention alignment and
  heteroscedastic uncertainty (NLL loss). Implemented as original code
  in `src/models/temporal_fusion.py`.

---

## Training-free reliability fallback (Deep Image Prior)

**Adopted from:** satellite-cloud-removal-dip
- Repo: https://github.com/strath-ai/satellite-cloud-removal-dip  (MIT)
- Concept adopted: per-image Deep Image Prior inpainting constrained
  by cloud mask, used as a reliability fallback. Implemented as original
  code in `src/evaluation/dip_fallback.py`.

---

## Benchmark datasets

| Dataset | Source | License | URL |
|---|---|---|---|
| SEN12MS-CR | Ebel et al. | CC BY 4.0 | https://mediatum.ub.tum.de/1554803 |
| SEN12MS-CR-TS | Ebel et al. | CC BY 4.0 | https://github.com/PatrickTUM/SEN12MS-CR-TS |
| CUHK-CR1/CR2 | Li et al. | Research use | https://github.com/littlebeen/DDPM-Enhancement-for-Cloud-Removal |

---

*Last updated: 2026-09-16*
