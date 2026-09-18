import math
import numpy as np
import pytest
import torch

from src.training.augmentations import (
    apply_sdaa,
    SunlightDirectionAwareAugmentation,
)


class TestSDAA:
    def test_sdaa_shape_preservation(self):
        b, c, h, w = 2, 3, 64, 64
        image = torch.rand(b, c, h, w)
        cloud_mask = (torch.rand(b, 1, h, w) > 0.7).float()

        augmented, shadow_mask = apply_sdaa(image, cloud_mask)
        assert augmented.shape == (b, c, h, w)
        assert shadow_mask.shape == (b, 1, h, w)
        assert (augmented >= 0.0).all() and (augmented <= 1.0).all()

    def test_solar_vector_shift_direction(self):
        """When solar azimuth is 90° (East, pi/2), shadows must cast strictly in the positive X direction."""
        b, c, h, w = 1, 1, 64, 64
        image = torch.ones(b, c, h, w) * 0.8
        cloud_mask = torch.zeros(b, 1, h, w)
        cloud_mask[:, :, 30:34, 20:24] = 1.0  # Centered cloud

        zenith = math.radians(45)
        azimuth = math.radians(90)  # East

        augmented, shadow_mask = apply_sdaa(
            image, cloud_mask,
            solar_zenith_rad=zenith,
            solar_azimuth_rad=azimuth,
            r_height=10.0,
            blur_sigma=0.0,
        )

        # Shadow should appear at x > 24
        assert shadow_mask[:, :, 30:34, 26:33].sum() > 0

    def test_gamma_darkening(self):
        """Shadowed pixels should be darker due to gamma factor (i^gamma < i for i in (0, 1))."""
        b, c, h, w = 1, 3, 32, 32
        image = torch.ones(b, c, h, w) * 0.6
        cloud_mask = torch.zeros(b, 1, h, w)
        cloud_mask[:, :, 10:14, 10:14] = 1.0

        augmented, shadow_mask = apply_sdaa(
            image, cloud_mask,
            solar_zenith_rad=math.radians(45),
            solar_azimuth_rad=math.radians(0),
            r_height=8.0,
            gamma=3.0,
        )

        # Where shadow mask > 0.5, augmented image must be darker than 0.6
        in_shadow = shadow_mask > 0.5
        if in_shadow.any():
            assert (augmented[in_shadow.expand(-1, 3, -1, -1)] < 0.6).all()

    def test_module_wrapper(self):
        sdaa = SunlightDirectionAwareAugmentation(p=1.0)
        sdaa.train()
        image = torch.rand(2, 3, 32, 32)
        cloud_mask = (torch.rand(2, 1, 32, 32) > 0.5).float()

        out_img, shadow = sdaa(image, cloud_mask)
        assert out_img.shape == (2, 3, 32, 32)

        sdaa.eval()
        out_eval, _ = sdaa(image, cloud_mask)
        # In eval mode, should return untouched image
        assert torch.equal(out_eval, image)
