import numpy as np


class TestNDVI:
    def test_ndvi_shape(self):
        from src.evaluation.analysis_readiness import ndvi
        image = np.random.rand(30, 30, 3).astype(np.float32)
        result = ndvi(image)
        assert result.shape == (30, 30)

    def test_ndvi_range_vegetation(self):
        from src.evaluation.analysis_readiness import ndvi
        image = np.zeros((10, 10, 3), dtype=np.float32)
        image[..., 1] = 0.1  # red
        image[..., 2] = 0.8  # nir
        ndvi_val = ndvi(image)
        assert np.all(ndvi_val > 0.5)

    def test_ndvi_range_bare_soil(self):
        from src.evaluation.analysis_readiness import ndvi
        image = np.zeros((10, 10, 3), dtype=np.float32)
        image[..., 1] = 0.4
        image[..., 2] = 0.4
        ndvi_val = ndvi(image)
        assert np.all(np.abs(ndvi_val) < 0.1)


class TestNDVIPreservation:
    def test_identical_returns_one(self):
        from src.evaluation.analysis_readiness import ndvi_preservation
        img = np.random.rand(30, 30, 3).astype(np.float32)
        score = ndvi_preservation(img, img)
        assert np.isclose(score, 1.0, atol=0.01)

    def test_opposite_returns_low(self):
        from src.evaluation.analysis_readiness import ndvi_preservation
        ref = np.zeros((10, 10, 3), dtype=np.float32)
        ref[..., 1] = 0.1
        ref[..., 2] = 0.8
        bad = np.zeros((10, 10, 3), dtype=np.float32)
        bad[..., 1] = 0.8
        bad[..., 2] = 0.1
        score = ndvi_preservation(bad, ref)
        assert score < 0.5

    def test_output_range(self):
        from src.evaluation.analysis_readiness import ndvi_preservation
        ref = np.random.rand(20, 20, 3).astype(np.float32)
        corr = np.random.rand(20, 20, 3).astype(np.float32)
        score = ndvi_preservation(corr, ref)
        assert 0.0 <= score <= 1.0


class TestStructuralSimilarity:
    def test_identical_returns_one(self):
        from src.evaluation.analysis_readiness import structural_similarity
        x = np.random.rand(30, 30, 3).astype(np.float32)
        score = structural_similarity(x, x)
        assert np.isclose(score, 1.0, atol=0.05)

    def test_output_range(self):
        from src.evaluation.analysis_readiness import structural_similarity
        x = np.random.rand(20, 20, 3).astype(np.float32)
        y = np.random.rand(20, 20, 3).astype(np.float32)
        score = structural_similarity(x, y)
        assert -1.0 <= score <= 1.0


class TestComputeARS:
    def test_returns_dict_with_keys(self):
        from src.evaluation.analysis_readiness import compute_ars
        conf = np.full((20, 20), 0.8, dtype=np.float32)
        result = compute_ars(conf)
        assert "ars" in result
        assert "components" in result
        assert "weights" in result

    def test_default_weights(self):
        from src.evaluation.analysis_readiness import compute_ars
        conf = np.full((20, 20), 1.0, dtype=np.float32)
        result = compute_ars(conf)
        assert np.isclose(result["ars"], 0.4, atol=0.01)

    def test_with_images(self):
        from src.evaluation.analysis_readiness import compute_ars
        conf = np.full((20, 20), 1.0, dtype=np.float32)
        img = np.random.rand(20, 20, 3).astype(np.float32)
        result = compute_ars(conf, img, img)
        assert 0.0 <= result["ars"] <= 1.0
        assert result["components"]["ndvi_preservation"] > 0.5
        assert result["components"]["structural_similarity"] > 0.5

    def test_ars_inverse_relationship_with_clouds(self):
        from src.evaluation.analysis_readiness import compute_ars
        ref = np.random.rand(20, 20, 3).astype(np.float32)
        good_corrected = ref.copy()
        bad_corrected = np.random.rand(20, 20, 3).astype(np.float32)
        good_conf = np.full((20, 20), 0.9, dtype=np.float32)
        bad_conf = np.full((20, 20), 0.1, dtype=np.float32)

        good_result = compute_ars(good_conf, good_corrected, ref)
        bad_result = compute_ars(bad_conf, bad_corrected, ref)
        assert good_result["ars"] >= bad_result["ars"]


class TestAnalysisReadiness:
    def test_evaluate_returns_dict(self):
        from src.evaluation.analysis_readiness import AnalysisReadiness
        ar = AnalysisReadiness()
        conf = np.full((20, 20), 0.8, dtype=np.float32)
        result = ar.evaluate(conf)
        assert "ars" in result

    def test_grade_thresholds(self):
        from src.evaluation.analysis_readiness import AnalysisReadiness
        ar = AnalysisReadiness()
        assert ar.grade(0.95) == "A"
        assert ar.grade(0.80) == "B"
        assert ar.grade(0.60) == "C"
        assert ar.grade(0.30) == "D"

    def test_custom_weights(self):
        from src.evaluation.analysis_readiness import AnalysisReadiness
        ar = AnalysisReadiness(weight_confidence=1.0, weight_ndvi=0.0, weight_structural=0.0)
        conf = np.full((20, 20), 0.75, dtype=np.float32)
        result = ar.evaluate(conf)
        assert np.isclose(result["ars"], 0.75)
