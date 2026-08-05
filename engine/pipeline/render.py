import cv2
import numpy as np
from engine.pipeline.warp import compute_perspective_matrix, warp_design_to_surface
from engine.pipeline.displacement import apply_displacement_map
from engine.pipeline.blend import apply_photometric_blending, match_histogram_lab
from engine.pipeline.mask import clean_binary_mask, feather_mask_edges, blend_with_alpha_mask

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
    linear_blend: bool = False
) -> np.ndarray:
    """
    Orchestrates the entire mockup render pipeline.
    1. Fits/scales/rotates the design image based on transform_params if supplied.
    2. Performs geometric alignment (homography warping) to target corners.
    3. Performs surface displacement mapping to deform design pixels.
    4. Applies lighting overlays (photometric blending) and color corrections.
    5. Feathers and anti-aliases the design zone boundary mask.
    6. Blends the design seamlessly on top of the product photo base.
    """
    h_base, w_base = base_image.shape[:2]

    # 1. Preprocess/transform the design image first if interactive parameters are provided
    design_processed = design_image.copy()

    # Ensure BGRA format
    if design_processed.shape[2] == 3:
        design_processed = cv2.cvtColor(design_processed, cv2.COLOR_BGR2BGRA)

    if transform_params:
        # Scale, rotate, offset the design relative to its own center
        scale = transform_params.get("scale", 1.0)
        rotation = transform_params.get("rotation", 0.0) # degrees
        offset_x = transform_params.get("offset_x", 0)
        offset_y = transform_params.get("offset_y", 0)

        h_ds, w_ds = design_processed.shape[:2]
        center = (w_ds // 2, h_ds // 2)

        # Get rotation & scaling matrix
        rot_mat = cv2.getRotationMatrix2D(center, rotation, scale)
        # Apply translation (offset)
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

    # 2. Geometric alignment (homography perspective transform)
    # If corners are not provided, we span the entire base image boundaries
    if not src_corners or not dst_corners:
        h_ds, w_ds = design_processed.shape[:2]
        src_corners = [[0, 0], [w_ds, 0], [w_ds, h_ds], [0, h_ds]]
        dst_corners = [[0, 0], [w_base, 0], [w_base, h_base], [0, h_base]]

    matrix = compute_perspective_matrix(src_corners, dst_corners)
    warped_design = warp_design_to_surface(design_processed, matrix, (w_base, h_base))

    # 3. Optional surface displacement mapping (for clothes, fabrics, wrinkles)
    if displacement_image is not None:
        if len(displacement_image.shape) == 3:
            displacement_gray = cv2.cvtColor(displacement_image, cv2.COLOR_BGR2GRAY)
        else:
            displacement_gray = displacement_image

        warped_design = apply_displacement_map(warped_design, displacement_gray, fold_intensity)

    # 4. Optional color correction matching and photometric blending
    if color_correct:
        warped_design = match_histogram_lab(warped_design, base_image)

    if lighting_image is not None:
        warped_design = apply_photometric_blending(warped_design, lighting_image, blend_mode)

    # 5. Mask processing, clean up and edge feathering
    if len(mask_image.shape) == 3:
        mask_gray = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
    else:
        mask_gray = mask_image

    cleaned_mask = clean_binary_mask(mask_gray)
    feathered_mask = feather_mask_edges(cleaned_mask, feather_radius)

    # 6. Final compositing of the warped design over base image
    final_composite = blend_with_alpha_mask(warped_design, base_image, feathered_mask, linear_blend=linear_blend)

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
