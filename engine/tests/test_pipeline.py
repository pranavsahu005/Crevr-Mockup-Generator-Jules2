import numpy as np
import cv2
import pytest
from engine.pipeline.warp import compute_perspective_matrix, warp_design_to_surface
from engine.pipeline.displacement import apply_displacement_map
from engine.pipeline.blend import apply_photometric_blending, match_histogram_lab
from engine.pipeline.mask import clean_binary_mask, feather_mask_edges, blend_with_alpha_mask
from engine.pipeline.render import render_mockup_pipeline, calculate_export_dimensions

def test_warp_pipeline():
    # Test computation of perspective homography matrix and warping
    src_pts = [[0, 0], [100, 0], [100, 100], [0, 100]]
    dst_pts = [[10, 10], [90, 5], [95, 95], [5, 90]]
    matrix = compute_perspective_matrix(src_pts, dst_pts)
    assert matrix.shape == (3, 3)

    design = np.ones((100, 100, 3), dtype=np.uint8) * 255
    warped = warp_design_to_surface(design, matrix, (100, 100))
    assert warped.shape == (100, 100, 4) # Output should have alpha channel

def test_displacement_pipeline():
    # Test per-pixel offset mapping using flat displacement map
    image = np.ones((50, 50, 4), dtype=np.uint8) * 128
    disp_map = np.ones((50, 50), dtype=np.uint8) * 128 # neutral height
    displaced = apply_displacement_map(image, disp_map, fold_intensity=10.0)
    assert displaced.shape == (50, 50, 4)

def test_blending_modes():
    # Test Multiply, Overlay, and Linear Burn blends
    design_warped = np.ones((20, 20, 4), dtype=np.uint8) * 200
    lighting_map = np.ones((20, 20), dtype=np.uint8) * 128 # half-brightness shadow overlay

    # Multiply should darken the output
    blended_mult = apply_photometric_blending(design_warped, lighting_map, "multiply")
    assert blended_mult[0, 0, 0] < 200
    assert blended_mult.shape == (20, 20, 4)

    # Overlay
    blended_overlay = apply_photometric_blending(design_warped, lighting_map, "overlay")
    assert blended_overlay.shape == (20, 20, 4)

def test_mask_feathering():
    # Test mask cleanup and edge feathering transitions
    mask = np.zeros((30, 30), dtype=np.uint8)
    cv2.rectangle(mask, (5, 5), (25, 25), 255, -1)

    cleaned = clean_binary_mask(mask)
    assert np.any(cleaned > 0)

    feathered = feather_mask_edges(cleaned, feather_radius=3)
    assert feathered.shape == (30, 30)
    assert feathered.dtype == np.float32
    assert np.max(feathered) <= 1.0

def test_end_to_end_pipeline():
    # Create simple mock base, design, and mask images
    base = np.zeros((100, 100, 3), dtype=np.uint8)
    design = np.ones((50, 50, 3), dtype=np.uint8) * 128
    mask = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(mask, (25, 25), (75, 75), 255, -1)

    # Run full mockup compositing render pipeline
    result = render_mockup_pipeline(
        base_image=base,
        design_image=design,
        mask_image=mask,
        fold_intensity=5.0,
        feather_radius=3,
        color_correct=False
    )
    assert result.shape == (100, 100, 3)

def test_export_dimensions():
    # Test millimeter to pixel DPI conversion
    dims = calculate_export_dimensions((101.6, 152.4), 300) # 4x6 inches at 300 DPI
    assert dims == (1200, 1800)

def test_metadata_encoding_and_existence():
    # Verify that all template metadata.json files exist, are UTF-8 encoded, and readable
    import json
    import os

    template_ids = ["laptop_01", "tshirt_01", "tshirt_02"]
    for t_id in template_ids:
        path = os.path.join("templates", t_id, "metadata.json")
        assert os.path.exists(path), f"Metadata file for {t_id} does not exist!"

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert data["id"] == t_id
            assert "category" in data
            assert "base_image" in data
            assert "mask_image" in data
            assert "design_zone_corners" in data
            # Ensure the label contains unicode characters if applicable and loads correctly
            assert isinstance(data["label"], str)
