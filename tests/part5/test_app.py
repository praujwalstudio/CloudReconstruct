import numpy as np


class TestNormalizeDisplay:
    def test_uint16_normalized(self):
        from src.app.app import normalize_display
        img = np.random.randint(0, 65535, (32, 32, 3), dtype=np.uint16)
        result = normalize_display(img)
        assert result.dtype == np.float64
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_uint8_normalized(self):
        from src.app.app import normalize_display
        img = np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8)
        result = normalize_display(img)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_uniform_image(self):
        from src.app.app import normalize_display
        img = np.ones((16, 16, 3), dtype=np.uint16) * 500
        result = normalize_display(img)
        assert result.min() >= -1e-6
        assert result.max() <= 1.0

    def test_float_input(self):
        from src.app.app import normalize_display
        img = np.random.rand(16, 16, 3).astype(np.float32)
        result = normalize_display(img)
        assert result.min() >= 0.0
        assert result.max() <= 1.0


class TestAppImport:
    def test_module_importable(self):
        import src.app.app
        assert src.app.app is not None

    def test_get_inference_model(self):
        from src.app.app import get_inference_model
        model = get_inference_model()
        assert model is not None
