"""Unit Tests for Accuracy Benchmarking & SOTA Validation Engine"""

import math
from pathlib import Path
import numpy as np
import pytest

from src.evaluation.benchmark_accuracy import (
    compute_rmse,
    compute_mae,
    compute_psnr,
    compute_ssim,
    compute_sam,
    compute_all_sota_metrics,
    SOTABenchmarkEngine,
    PUBLISHED_MULTI_TEMPORAL_BASELINES,
    PUBLISHED_MONO_TEMPORAL_BASELINES,
)


class TestSOTAMetrics:
    def test_compute_rmse(self):
        a = np.zeros((10, 10, 3), dtype=np.float32)
        b = np.ones((10, 10, 3), dtype=np.float32) * 0.1
        rmse = compute_rmse(a, b)
        assert np.isclose(rmse, 0.1)

    def test_compute_mae(self):
        a = np.zeros((10, 10, 3), dtype=np.float32)
        b = np.ones((10, 10, 3), dtype=np.float32) * 0.05
        mae = compute_mae(a, b)
        assert np.isclose(mae, 0.05)

    def test_compute_psnr_formula(self):
        a = np.zeros((10, 10, 3), dtype=np.float32)
        b = np.ones((10, 10, 3), dtype=np.float32) * 0.1  # RMSE = 0.1 -> PSNR = 20 dB
        psnr_val = compute_psnr(a, b, data_range=1.0)
        assert np.isclose(psnr_val, 20.0, atol=0.1)

    def test_compute_ssim(self):
        a = np.random.uniform(0, 1, (32, 32, 3)).astype(np.float32)
        ssim_val = compute_ssim(a, a, data_range=1.0)
        assert np.isclose(ssim_val, 1.0, atol=1e-3)

    def test_compute_sam(self):
        # Perfectly aligned spectral vectors should have angle 0.0 degrees
        a = np.ones((10, 10, 3), dtype=np.float32) * 0.2
        b = np.ones((10, 10, 3), dtype=np.float32) * 0.8
        sam_val = compute_sam(a, b)
        assert np.isclose(sam_val, 0.0, atol=1e-3)

    def test_compute_all_sota_metrics(self):
        a = np.random.uniform(0, 1, (16, 16, 3)).astype(np.float32)
        b = np.random.uniform(0, 1, (16, 16, 3)).astype(np.float32)
        metrics = compute_all_sota_metrics(a, b)
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "psnr" in metrics
        assert "ssim" in metrics
        assert "sam" in metrics
        assert metrics["rmse"] >= 0.0
        assert metrics["mae"] >= 0.0


class TestSOTABenchmarkEngine:
    def test_baselines_present(self):
        assert "UnCRtainTS (Ebel SOTA)" in PUBLISHED_MULTI_TEMPORAL_BASELINES
        assert "DSen2-CR" in PUBLISHED_MULTI_TEMPORAL_BASELINES
        assert "GLF-CR" in PUBLISHED_MONO_TEMPORAL_BASELINES
        assert "UnCRtainTS (SOTA)" in PUBLISHED_MONO_TEMPORAL_BASELINES

    def test_generate_markdown_report(self):
        engine = SOTABenchmarkEngine(device="cpu")
        sample_multi = {"rmse": 0.052, "psnr": 26.85, "ssim": 0.8350, "sam": 10.950}
        sample_mono = {"mae": 0.0305, "psnr": 27.95, "ssim": 0.8750, "sam": 9.200}
        report = engine.generate_markdown_report(sample_multi, sample_mono)
        assert "Target A: Multi-Temporal Reconstruction" in report
        assert "Target B: Mono-Temporal Reconstruction" in report
        assert "UnCRtainTS" in report
        assert "26.85" in report
        assert "27.95" in report
