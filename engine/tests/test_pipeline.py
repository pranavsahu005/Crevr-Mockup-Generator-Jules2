import numpy as np
import cv2
import pytest
from engine.pipeline.warp import compute_perspective_matrix, warp_design_to_surface, apply_corner_radius, apply_perspective_bend
from engine.pipeline.displacement import apply_displacement_map
from engine.pipeline.blend import apply_photometric_blending, match_histogram_lab, recolor_garment
from engine.pipeline.mask import clean_binary_mask, feather_mask_edges, blend_with_alpha_mask, remove_solid_background
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

# ==================== Phase 2 Unit Tests ====================

def test_garment_recoloring():
    # White base image (representing a white fabric mockup t-shirt)
    base = np.ones((50, 50, 3), dtype=np.uint8) * 200

    # Recolor to custom orange HEX code (#FF5733)
    recolored = recolor_garment(base, "#FF5733")
    assert recolored.shape == (50, 50, 3)
    # The image should no longer be standard gray/white
    b, g, r = cv2.split(recolored)
    assert not np.array_equal(r, b)

def test_background_removal():
    # Design image with a solid solid black background at corners
    design = np.zeros((40, 40, 3), dtype=np.uint8)
    # Add a white central logo rectangle
    cv2.rectangle(design, (10, 10), (30, 30), (255, 255, 255), -1)

    # Strip black backdrop
    transparent_design = remove_solid_background(design, tolerance=30)
    assert transparent_design.shape == (40, 40, 4)
    # Corner alpha should be transparent (0)
    assert transparent_design[0, 0, 3] == 0
    # Central logo alpha should remain opaque (255)
    assert transparent_design[20, 20, 3] > 200

def test_corner_radius():
    design = np.ones((50, 50, 4), dtype=np.uint8) * 255
    rounded = apply_corner_radius(design, radius_px=10.0)
    assert rounded.shape == (50, 50, 4)
    # The extreme corners of the image should now be transparent
    assert rounded[0, 0, 3] == 0
    # The center of the image should remain opaque
    assert rounded[25, 25, 3] == 255

def test_perspective_bend():
    # Quad corners: [tl, tr, br, bl]
    corners = [[10, 10], [90, 10], [90, 90], [10, 90]]

    # Narrow the top edge
    narrowed = apply_perspective_bend(corners, bend_factor=-20.0)
    # tl.x should move right, tr.x should move left
    assert narrowed[0][0] > 10
    assert narrowed[1][0] < 90

    # Widen the top edge
    widened = apply_perspective_bend(corners, bend_factor=20.0)
    # tl.x should move left, tr.x should move right
    assert widened[0][0] < 10
    assert widened[1][0] > 90
