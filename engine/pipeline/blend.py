import cv2
import numpy as np

def hex_to_bgr(hex_str: str) -> tuple:
    """Converts HEX color string to (B, G, R) tuple."""
    hex_str = hex_str.lstrip('#')
    if len(hex_str) == 3:
        hex_str = "".join([c*2 for c in hex_str])
    r = int(hex_str[0:2], 16)
    g = int(hex_str[2:4], 16)
    b = int(hex_str[4:6], 16)
    return (b, g, r)

def recolor_garment(base_image: np.ndarray, hex_color: str) -> np.ndarray:
    """
    Recolors the white-ish fabric of apparel base images using target HEX color.
    Keeps shadows, creases, and lighting of the fabric intact using LAB space luminance scaling.
    """
    if not hex_color:
        return base_image
    try:
        b_tgt, g_tgt, r_tgt = hex_to_bgr(hex_color)
    except Exception as e:
        print(f"Failed to parse garment color {hex_color}: {e}")
        return base_image

    # Create target pixel in BGR
    tgt_pixel = np.array([[[b_tgt, g_tgt, r_tgt]]], dtype=np.uint8)
    tgt_lab = cv2.cvtColor(tgt_pixel, cv2.COLOR_BGR2LAB).astype(np.float32)[0, 0]
    l_tgt, a_tgt, b_tgt_lab = tgt_lab[0], tgt_lab[1], tgt_lab[2]

    # Identify white/light neutral fabric pixels
    gray = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)
    is_light = gray > 110

    b_ch, g_ch, r_ch = cv2.split(base_image)
    is_neutral = (
        (np.abs(r_ch.astype(np.float32) - g_ch.astype(np.float32)) < 35) &
        (np.abs(g_ch.astype(np.float32) - b_ch.astype(np.float32)) < 35) &
        (np.abs(r_ch.astype(np.float32) - b_ch.astype(np.float32)) < 35)
    )

    fabric_mask = is_light & is_neutral
    if not np.any(fabric_mask):
        return base_image

    # Convert base image to LAB
    result_lab = cv2.cvtColor(base_image, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Recolor only the fabric mask pixels:
    # 1. Update Chrominance (A & B channels) to target
    result_lab[fabric_mask, 1] = a_tgt
    result_lab[fabric_mask, 2] = b_tgt_lab

    # 2. Modulate Luminance (L channel) to preserve highlights and shadows
    # High-quality modulation factor: scale by target L relative to standard white luminance (~220)
    l_scale = l_tgt / 220.0
    l_scale = max(l_scale, 0.1) # keep a floor to prevent pure pitch-black flatness

    result_lab[fabric_mask, 0] = result_lab[fabric_mask, 0] * l_scale
    result_lab[:, :, 0] = np.clip(result_lab[:, :, 0], 0, 255)

    # Convert back to BGR
    result_bgr = cv2.cvtColor(result_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    # Soften transition at mask edges slightly
    mask_blurred = cv2.GaussianBlur(fabric_mask.astype(np.float32) * 255.0, (5, 5), 0) / 255.0
    mask_3ch = np.stack([mask_blurred] * 3, axis=2)

    final_image = (result_bgr.astype(np.float32) * mask_3ch + base_image.astype(np.float32) * (1.0 - mask_3ch)).astype(np.uint8)
    return final_image

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
