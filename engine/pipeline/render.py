import cv2
import numpy as np
from engine.pipeline.warp import compute_perspective_matrix, warp_design_to_surface, apply_corner_radius, apply_perspective_bend
from engine.pipeline.displacement import apply_displacement_map
from engine.pipeline.blend import apply_photometric_blending, match_histogram_lab, recolor_garment
from engine.pipeline.mask import clean_binary_mask, feather_mask_edges, blend_with_alpha_mask, remove_solid_background

def render_mockup_pipeline(
    base_image: np.ndarray,
    design_image: np.ndarray,
    mask_image: np.ndarray,
    displacement_image: np.ndarray = None,
    lighting_image: np.ndarray = None,
    src_corners: list = None,
    dst_corners: list = None,
    fold_intensity: float = 15.0,
    feather_radius: int = 5,
    blend_mode: str = "multiply",
    color_correct: bool = True,
    transform_params: dict = None,

    # Phase 2 Parameters
    garment_color: str = None,
    corner_radius: float = 0.0,
    perspective_bend: float = 0.0,
    remove_background: bool = False
) -> np.ndarray:
    """
    Orchestrates the entire mockup render pipeline, fully updated for Phase 2.
    1. Removes background (chroma-keying) if requested.
    2. Applies rounded corners (corner radius) to the design.
    3. Fits/scales/rotates the design image based on transform_params.
    4. Adjusts destination corners with perspective tilt/bend.
    5. Performs geometric alignment (homography warping) to target corners.
    6. Performs surface displacement mapping to deform design pixels.
    7. Applies garment color customization directly to base_image.
    8. Applies lighting overlays (photometric blending) and color corrections.
    9. Feathers and anti-aliases the design zone boundary mask.
    10. Blends the design seamlessly on top of the product photo base.
    """
    h_base, w_base = base_image.shape[:2]

    # 1. Automatic background removal
    design_processed = design_image.copy()
    if remove_background:
        design_processed = remove_solid_background(design_processed)

    # Ensure BGRA format
    if design_processed.shape[2] == 3:
        design_processed = cv2.cvtColor(design_processed, cv2.COLOR_BGR2BGRA)

    # 2. Corner rounding
    if corner_radius > 0.0:
        design_processed = apply_corner_radius(design_processed, corner_radius)

    # 3. Preprocess/transform design based on interactive parameters if provided
    if transform_params:
        scale = transform_params.get("scale", 1.0)
        rotation = transform_params.get("rotation", 0.0) # degrees
        offset_x = transform_params.get("offset_x", 0)
        offset_y = transform_params.get("offset_y", 0)

        h_ds, w_ds = design_processed.shape[:2]
        center = (w_ds // 2, h_ds // 2)

        # Get rotation & scaling matrix
        rot_mat = cv2.getRotationMatrix2D(center, rotation, scale)
        rot_mat[0, 2] += offset_x
        rot_mat[1, 2] += offset_y

        design_processed = cv2.warpAffine(
            design_processed,
            rot_mat,
            (w_ds, h_ds),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0)
        )

    # 4. Geometric alignment (homography perspective transform)
    if not src_corners or not dst_corners:
        h_ds, w_ds = design_processed.shape[:2]
        src_corners = [[0, 0], [w_ds, 0], [w_ds, h_ds], [0, h_ds]]
        dst_corners = [[0, 0], [w_base, 0], [w_base, h_base], [0, h_base]]

    # Apply dynamic perspective tilt/bend adjustment
    if perspective_bend != 0.0:
        dst_corners = apply_perspective_bend(dst_corners, perspective_bend)

    matrix = compute_perspective_matrix(src_corners, dst_corners)
    warped_design = warp_design_to_surface(design_processed, matrix, (w_base, h_base))

    # 5. Surface displacement mapping (for clothes, fabrics, wrinkles)
    if displacement_image is not None:
        if len(displacement_image.shape) == 3:
            displacement_gray = cv2.cvtColor(displacement_image, cv2.COLOR_BGR2GRAY)
        else:
            displacement_gray = displacement_image

        warped_design = apply_displacement_map(warped_design, displacement_gray, fold_intensity)

    # 6. Garment recoloring
    recolored_base = base_image.copy()
    if garment_color:
        recolored_base = recolor_garment(recolored_base, garment_color)

    # 7. Color correction matching and photometric blending
    if color_correct:
        warped_design = match_histogram_lab(warped_design, recolored_base)

    if lighting_image is not None:
        warped_design = apply_photometric_blending(warped_design, lighting_image, blend_mode)

    # 8. Mask processing, clean up and edge feathering
    if len(mask_image.shape) == 3:
        mask_gray = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
    else:
        mask_gray = mask_image

    cleaned_mask = clean_binary_mask(mask_gray)
    feathered_mask = feather_mask_edges(cleaned_mask, feather_radius)

    # 9. Final compositing of the warped design over base image
    final_composite = blend_with_alpha_mask(warped_design, recolored_base, feathered_mask)

    return final_composite

def calculate_export_dimensions(physical_size_mm: tuple, dpi: int) -> tuple:
    """
    Computes required pixel dimensions for print based on physical size in mm and target DPI.
    physical_size_mm: (width_mm, height_mm).
    Returns (width_pixels, height_pixels).
    """
    width_mm, height_mm = physical_size_mm
    width_in = width_mm / 25.4
    height_in = height_mm / 25.4
    return (int(round(width_in * dpi)), int(round(height_in * dpi)))
