import numpy as np
import torch
from pathlib import Path


class TestCloudFreeInference:
    def test_import(self):
        from src.evaluation.inference import CloudFreeInference
        assert CloudFreeInference is not None

    def test_init_without_checkpoints(self):
        from src.evaluation.inference import CloudFreeInference
        model = CloudFreeInference(device="cpu")
        assert model.pipeline is not None
        assert model.device == "cpu"

    def test_correct_returns_dict(self):
        from src.evaluation.inference import CloudFreeInference
        model = CloudFreeInference(device="cpu")
        image = np.random.randint(0, 1023, (64, 64, 3), dtype=np.uint16)
        result = model.correct(image)
        assert isinstance(result, dict)
        assert "corrected" in result
        assert "density" in result
        assert "confidence" in result
        assert "ars" in result

    def test_correct_output_shapes(self):
        from src.evaluation.inference import CloudFreeInference
        model = CloudFreeInference(device="cpu")
        h, w = 64, 64
        image = np.random.randint(0, 1023, (h, w, 3), dtype=np.uint16)
        result = model.correct(image)
        assert result["corrected"].shape == (h, w, 3)
        assert result["density"].shape == (h, w)
        assert result["confidence"].shape == (h, w)

    def test_correct_output_dtype_preserved(self):
        from src.evaluation.inference import CloudFreeInference
        model = CloudFreeInference(device="cpu")
        image = np.random.randint(0, 1023, (64, 64, 3), dtype=np.uint16)
        result = model.correct(image)
        assert result["corrected"].dtype == np.uint16

    def test_correct_ars_has_expected_keys(self):
        from src.evaluation.inference import CloudFreeInference
        model = CloudFreeInference(device="cpu")
        image = np.random.randint(0, 1023, (64, 64, 3), dtype=np.uint16)
        result = model.correct(image)
        assert "ars" in result["ars"]
        assert "components" in result["ars"]
        assert "weights" in result["ars"]

    def test_correct_density_in_range(self):
        from src.evaluation.inference import CloudFreeInference
        model = CloudFreeInference(device="cpu")
        image = np.random.randint(0, 1023, (64, 64, 3), dtype=np.uint16)
        result = model.correct(image)
        assert result["density"].min() >= 0.0
        assert result["density"].max() <= 1.0

    def test_correct_confidence_in_range(self):
        from src.evaluation.inference import CloudFreeInference
        model = CloudFreeInference(device="cpu")
        image = np.random.randint(0, 1023, (64, 64, 3), dtype=np.uint16)
        result = model.correct(image)
        assert result["confidence"].min() >= 0.0
        assert result["confidence"].max() <= 1.0

    def test_correct_with_sar(self):
        from src.evaluation.inference import CloudFreeInference
        model = CloudFreeInference(device="cpu")
        image = np.random.randint(0, 1023, (64, 64, 3), dtype=np.uint16)
        sar = np.random.randint(0, 255, (64, 64, 2), dtype=np.uint8)
        result = model.correct(image, sar=sar)
        assert result["corrected"].shape == (64, 64, 3)

    def test_correct_and_save_creates_file(self, tmp_path):
        from src.evaluation.inference import CloudFreeInference
        model = CloudFreeInference(device="cpu")
        image = np.random.randint(0, 1023, (32, 32, 3), dtype=np.uint16)
        out_path = tmp_path / "result.tif"
        result_path = model.correct_and_save(out_path, image)
        assert result_path.exists()
        assert result_path.suffix == ".tif"

    def test_correct_uint8_input(self):
        from src.evaluation.inference import CloudFreeInference
        model = CloudFreeInference(device="cpu")
        image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        result = model.correct(image)
        assert result["corrected"].dtype == np.uint8


class TestNumpyTensorConversion:
    def test_numpy_to_tensor_uint16(self):
        from src.evaluation.inference import _numpy_to_tensor
        img = np.random.randint(0, 65535, (32, 32, 3), dtype=np.uint16)
        tensor = _numpy_to_tensor(img, "cpu")
        assert tensor.shape == (1, 3, 32, 32)
        assert tensor.dtype == torch.float32
        assert tensor.min() >= 0.0
        assert tensor.max() <= 1.0

    def test_numpy_to_tensor_uint8(self):
        from src.evaluation.inference import _numpy_to_tensor
        img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        tensor = _numpy_to_tensor(img, "cpu")
        assert tensor.max() <= 1.0

    def test_tensor_to_numpy_uint16(self):
        from src.evaluation.inference import _tensor_to_numpy
        import torch
        tensor = torch.rand(1, 3, 32, 32)
        arr = _tensor_to_numpy(tensor, np.uint16)
        assert arr.shape == (32, 32, 3)
        assert arr.dtype == np.uint16


class TestLoadCheckpoint:
    def test_load_checkpoint_nonexistent(self):
        from src.evaluation.inference import _load_checkpoint
        result = _load_checkpoint(Path(r"C:\nonexistent\path.pt"))
        assert result is None

    def test_load_checkpoint_empty_dir(self, tmp_path):
        from src.evaluation.inference import _load_checkpoint
        result = _load_checkpoint(tmp_path / "empty")
        assert result is None
