import numpy as np
import cv2
import pytest
from engine.pipeline.mask import blend_with_alpha_mask
from engine.main import sniff_and_sanitize_image
from PIL import Image
import io

def test_blend_with_alpha_mask_linear():
    # Test that linear light blending works without error and differs from standard blending
    foreground = np.zeros((100, 100, 4), dtype=np.uint8)
    # Red foreground
    foreground[:, :, 0] = 0
    foreground[:, :, 1] = 0
    foreground[:, :, 2] = 255
    foreground[:, :, 3] = 255

    background = np.zeros((100, 100, 3), dtype=np.uint8)
    # Green background
    background[:, :, 0] = 0
    background[:, :, 1] = 255
    background[:, :, 2] = 0

    feathered_mask = np.ones((100, 100), dtype=np.float32) * 0.5 # 50% opacity

    standard_blend = blend_with_alpha_mask(foreground, background, feathered_mask, linear_blend=False)
    linear_blend = blend_with_alpha_mask(foreground, background, feathered_mask, linear_blend=True)

    # They should not be identical because linear blending operates in linear light space
    assert not np.array_equal(standard_blend, linear_blend)
    assert standard_blend.shape == (100, 100, 3)
    assert linear_blend.shape == (100, 100, 3)
