"""Unit Tests for Standardized Quality & ARS Report Generator"""

import json
from pathlib import Path
import numpy as np
import pytest

from src.evaluation.report_generator import QualityReportGenerator


class TestQualityReportGenerator:
    @pytest.fixture
    def report_gen(self, tmp_path):
        return QualityReportGenerator(output_dir=tmp_path)

    def test_calculate_metrics(self, report_gen):
        h, w = 128, 128
        density = np.zeros((h, w), dtype=np.float32)
        density[30:70, 30:70] = 0.8  # cloud region
        confidence = np.ones((h, w), dtype=np.float32) * 0.9
        cloudy = np.ones((h, w, 3), dtype=np.float32) * 0.7
        corrected = np.ones((h, w, 3), dtype=np.float32) * 0.3

        metrics = report_gen.calculate_metrics(
            image_id="test_scene_01",
            density_map=density,
            confidence_map=confidence,
            corrected_image=corrected,
            cloudy_image=cloudy,
        )

        assert metrics["image_id"] == "test_scene_01"
        assert "cloud_cover_pct" in metrics
        assert metrics["cloud_cover_pct"] > 0.0
        assert "recovered_surface_area_pct" in metrics
        assert "average_confidence_pct" in metrics
        assert "ars_score" in metrics
        assert metrics["quality_grade"] in ("A", "B", "C", "F")
        assert metrics["image_dimensions"]["height"] == h
        assert metrics["image_dimensions"]["width"] == w

    def test_export_json(self, report_gen, tmp_path):
        report_data = {
            "image_id": "scene_xyz",
            "cloud_cover_pct": 25.5,
            "ars_score": 0.88,
            "quality_grade": "A",
        }
        json_path = report_gen.export_json(report_data)
        assert json_path.exists()

        with open(json_path, "r") as f:
            loaded = json.load(f)
        assert loaded["image_id"] == "scene_xyz"
        assert loaded["quality_grade"] == "A"

    def test_export_pdf(self, report_gen, tmp_path):
        h, w = 64, 64
        density = np.random.uniform(0, 1, (h, w)).astype(np.float32)
        confidence = np.random.uniform(0.7, 1, (h, w)).astype(np.float32)
        cloudy = np.random.uniform(0, 1, (h, w, 3)).astype(np.float32)
        corrected = np.random.uniform(0, 1, (h, w, 3)).astype(np.float32)

        report_data = {
            "image_id": "pdf_test_scene",
            "timestamp": "2026-08-25T12:00:00",
            "cloud_cover_pct": 32.4,
            "recovered_surface_area_pct": 94.2,
            "average_confidence_pct": 89.1,
            "ndvi_preservation_score": 0.852,
            "ars_score": 0.892,
            "quality_grade": "A",
        }

        pdf_path = report_gen.export_pdf(
            report_data, cloudy, density, confidence, corrected
        )
        assert pdf_path.exists()
        assert pdf_path.stat().st_size > 1000  # valid non-empty PDF file

    def test_generate_report_full(self, report_gen):
        h, w = 64, 64
        density = np.zeros((h, w), dtype=np.float32)
        confidence = np.ones((h, w), dtype=np.float32)
        cloudy = np.ones((h, w, 3), dtype=np.float32)
        corrected = np.ones((h, w, 3), dtype=np.float32)

        res = report_gen.generate_report(
            "full_test_scene",
            density, confidence, corrected, cloudy
        )
        assert "json_report_path" in res
        assert "pdf_report_path" in res
        assert Path(res["json_report_path"]).exists()
        assert Path(res["pdf_report_path"]).exists()
