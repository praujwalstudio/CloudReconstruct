import numpy as np
import pytest
import json
from pathlib import Path
from src.preprocessing.patch_generator import PatchGenerator, PatchConfig, PatchMetadata


class TestPatchGeneratorInit:
    def test_default_config(self):
        gen = PatchGenerator()
        assert gen.config.patch_size == 256
        assert gen.config.stride == 128
        assert gen.config.bands == [0, 1, 2]

    def test_custom_config(self):
        cfg = PatchConfig(patch_size=128, stride=64)
        gen = PatchGenerator(cfg)
        assert gen.config.patch_size == 128
        assert gen.config.stride == 64


class TestExtractFromScene:
    @pytest.fixture
    def scene_data(self):
        image = np.random.randint(0, 500, (264, 264, 3), dtype=np.uint16)
        mask = np.zeros((264, 264), dtype=np.uint8)
        mask[50:150, 50:150] = 1
        density = mask.astype(np.float32)
        return image, mask, density

    def test_returns_list_of_tuples(self, scene_data):
        image, mask, density = scene_data
        gen = PatchGenerator(PatchConfig(patch_size=128, stride=64))
        patches = gen.extract_from_scene(image, mask, density, "test_scene", None)
        assert len(patches) > 0
        assert all(isinstance(p, tuple) and len(p) == 2 for p in patches)
        assert all(isinstance(p[0], np.ndarray) for p in patches)
        assert all(isinstance(p[1], PatchMetadata) for p in patches)

    def test_patch_shape(self, scene_data):
        image, mask, density = scene_data
        gen = PatchGenerator(PatchConfig(patch_size=128, stride=128))
        patches = gen.extract_from_scene(image, mask, density, "test_scene", None)
        for patch_data, _ in patches:
            assert patch_data.shape == (128, 128, 3)


class TestPatchMetadata:
    def test_default_split_empty(self):
        meta = PatchMetadata(
            scene="test", patch_id="p1",
            row=0, col=0, x=0, y=0,
            width=256, height=256,
            cloud_fraction=0.5, density_class=1,
        )
        assert meta.split == ""

    def test_serialization(self):
        meta = PatchMetadata(
            scene="test", patch_id="p1",
            row=0, col=0, x=0, y=0,
            width=256, height=256,
            cloud_fraction=0.5, density_class=2,
            split="train",
        )
        d = meta.__dict__
        assert d["scene"] == "test"
        assert d["split"] == "train"


class TestSplitPatches:
    def test_scene_level_split_no_leakage(self):
        gen = PatchGenerator()
        metas = []
        names = []
        for scene in ["scene_a", "scene_a", "scene_b", "scene_b", "scene_c", "scene_c"]:
            meta = PatchMetadata(
                scene=scene, patch_id=f"{scene}_p1",
                row=0, col=0, x=0, y=0,
                width=256, height=256,
                cloud_fraction=0.3, density_class=1,
            )
            metas.append(meta)
            names.append(scene)

        gen.split_patches(metas, names)

        # All patches from same scene must have the same split
        splits_by_scene = {}
        for meta in metas:
            if meta.scene not in splits_by_scene:
                splits_by_scene[meta.scene] = meta.split
            else:
                assert splits_by_scene[meta.scene] == meta.split


class TestSavePatch:
    def test_saves_npy_and_json(self, tmp_path):
        gen = PatchGenerator()
        data = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint16)
        meta = PatchMetadata(
            scene="test", patch_id="test_p0",
            row=0, col=0, x=0, y=0,
            width=64, height=64,
            cloud_fraction=0.5, density_class=1,
            split="train",
        )
        gen.save_patch(data, meta, tmp_path)

        npy_path = tmp_path / "train" / "test_p0.npy"
        json_path = tmp_path / "train" / "test_p0_meta.json"

        assert npy_path.exists()
        assert json_path.exists()

        loaded = np.load(str(npy_path))
        assert np.array_equal(loaded, data)

        with open(json_path) as f:
            loaded_meta = json.load(f)
            assert loaded_meta["patch_id"] == "test_p0"
            assert loaded_meta["split"] == "train"
