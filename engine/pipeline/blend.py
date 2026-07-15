import cv2
import numpy as np

def blend_multiply(base: np.ndarray, design: np.ndarray) -> np.ndarray:
    """
    Standard multiply blending: (Base * Design) / 255
    Expects float precision. Inputs: float32 arrays of same shape.
    """
    return (base * design) / 255.0

def blend_overlay(base: np.ndarray, design: np.ndarray) -> np.ndarray:
    """
    Standard overlay blending.
    If base < 128: 2 * base * design / 255
    Else: 255 - 2 * (255 - base) * (255 - design) / 255
    """
    mask = base < 128.0
    result = np.empty_like(base)
    result[mask] = (2.0 * base[mask] * design[mask]) / 255.0
    result[~mask] = 255.0 - (2.0 * (255.0 - base[~mask]) * (255.0 - design[~mask])) / 255.0
    return result

def blend_linear_burn(base: np.ndarray, design: np.ndarray) -> np.ndarray:
    """
    Linear Burn blending: Base + Design - 255
    """
    return np.clip(base + design - 255.0, 0.0, 255.0)

def apply_photometric_blending(
    design_warped: np.ndarray,
    lighting_map: np.ndarray,
    blend_mode: str = "multiply"
) -> np.ndarray:
    """
    Blends the warped design with the lighting/shadow map of the product photo.
    design_warped: BGRA image (uint8, 0-255).
    lighting_map: Grayscale (0-255) indicating the lighting overlay.
                  If BGR/BGRA, it will be converted or squeezed.
    blend_mode: 'multiply', 'overlay', 'linear_burn'.
    """
    # Ensure lighting map matches shape
    h, w = design_warped.shape[:2]
    if len(lighting_map.shape) == 3:
        if lighting_map.shape[2] == 4:
            lighting_map = cv2.cvtColor(lighting_map, cv2.COLOR_BGRA2GRAY)
        elif lighting_map.shape[2] == 3:
            lighting_map = cv2.cvtColor(lighting_map, cv2.COLOR_BGR2GRAY)

    if lighting_map.shape[:2] != (h, w):
        lighting_map = cv2.resize(lighting_map, (w, h), interpolation=cv2.INTER_LINEAR)

    # Separate color and alpha channels
    design_bgr = design_warped[:, :, :3].astype(np.float32)
    design_alpha = design_warped[:, :, 3].astype(np.float32)

    # Expand lighting map to 3 channels for channel-wise operations
    lighting_3ch = np.stack([lighting_map.astype(np.float32)] * 3, axis=2)

    # Apply selected blend mode
    mode = blend_mode.lower()
    if mode == "multiply":
        blended_bgr = blend_multiply(lighting_3ch, design_bgr)
    elif mode == "overlay":
        blended_bgr = blend_overlay(lighting_3ch, design_bgr)
    elif mode == "linear_burn":
        blended_bgr = blend_linear_burn(lighting_3ch, design_bgr)
    else:
        # Fallback to Multiply
        blended_bgr = blend_multiply(lighting_3ch, design_bgr)

    # Clip values to 0-255
    blended_bgr = np.clip(blended_bgr, 0.0, 255.0).astype(np.uint8)

    # Reassemble BGRA
    result = np.zeros_like(design_warped)
    result[:, :, :3] = blended_bgr
    result[:, :, 3] = design_alpha.astype(np.uint8)

    return result

def match_histogram_lab(design: np.ndarray, base_photo: np.ndarray) -> np.ndarray:
    """
    Adjusts the design color temperature and lightness to match the base photo
    using classical LAB histogram matching (mean & standard deviation alignment).
    design: BGRA image.
    base_photo: BGR/BGRA image of the template.
    """
    design_bgr = design[:, :, :3].copy()
    design_alpha = design[:, :, 3]

    base_bgr = base_photo[:, :, :3]

    # Convert to LAB color space
    design_lab = cv2.cvtColor(design_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    base_lab = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Compute mean and standard deviation for both
    d_mean, d_std = cv2.meanStdDev(design_lab)
    b_mean, b_std = cv2.meanStdDev(base_lab)

    d_mean = d_mean.flatten()
    d_std = d_std.flatten()
    b_mean = b_mean.flatten()
    b_std = b_std.flatten()

    # Align statistics per channel
    # Avoid division by zero
    for i in range(3):
        std_ratio = b_std[i] / (d_std[i] + 1e-6)
        # Apply standard deviation and mean shift
        # We clamp std_ratio to avoid extreme color warping
        std_ratio = np.clip(std_ratio, 0.2, 5.0)
        design_lab[:, :, i] = (design_lab[:, :, i] - d_mean[i]) * std_ratio + b_mean[i]

    # Clip and cast back
    design_lab = np.clip(design_lab, 0, 255).astype(np.uint8)
    matched_bgr = cv2.cvtColor(design_lab, cv2.COLOR_LAB2BGR)

    result = np.zeros_like(design)
    result[:, :, :3] = matched_bgr
    result[:, :, 3] = design_alpha

    return result
