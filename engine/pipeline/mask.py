import cv2
import numpy as np

def clean_binary_mask(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Cleans up noisy or jagged binary masks using Morphological Closing then Opening.
    mask: grayscale binary mask (values 0 and 255).
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    # Close small holes
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # Open to remove small stray specs
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    return opened

def feather_mask_edges(mask: np.ndarray, feather_radius: int) -> np.ndarray:
    """
    Creates a soft feathered edge around the mask using a Gaussian blur to avoid harsh outlines.
    mask: grayscale mask (0-255).
    feather_radius: radius of Gaussian blur (must be positive odd integer, or we adjust it).
    Returns a float32 mask normalized to [0, 1].
    """
    if feather_radius <= 0:
        return (mask / 255.0).astype(np.float32)

    # Ensure odd kernel size
    k_size = feather_radius * 2 + 1

    # Blur the mask
    blurred = cv2.GaussianBlur(mask.astype(np.float32), (k_size, k_size), 0)

    # Normalize to [0, 1] range
    feathered_mask = np.clip(blurred / 255.0, 0.0, 1.0)
    return feathered_mask

def blend_with_alpha_mask(
    foreground: np.ndarray,
    background: np.ndarray,
    feathered_mask: np.ndarray,
    linear_blend: bool = False
) -> np.ndarray:
    """
    Performs alpha blending of foreground over background using a feathered mask.
    foreground: BGRA or BGR image (uint8, same dimensions as background).
    background: BGR image.
    feathered_mask: 2D float32 mask of shape (H, W) with values [0, 1].
    linear_blend: perform blending in linear light space (Gamma 2.2 approximation).
    Returns the composited BGR image (uint8).
    """
    # Ensure 3-channel format for foreground
    if foreground.shape[2] == 4:
        # Incorporate the design's own embedded alpha channel if present
        design_alpha = (foreground[:, :, 3] / 255.0).astype(np.float32)
        combined_mask = feathered_mask * design_alpha
        fg_bgr = foreground[:, :, :3].astype(np.float32)
    else:
        combined_mask = feathered_mask
        fg_bgr = foreground.astype(np.float32)

    bg_bgr = background[:, :, :3].astype(np.float32)

    # Expand combined mask to 3 channels for broadcasting
    mask_3ch = np.stack([combined_mask] * 3, axis=2)

    # Composite: FG * mask + BG * (1 - mask)
    if linear_blend:
        fg_linear = (fg_bgr / 255.0) ** 2.2
        bg_linear = (bg_bgr / 255.0) ** 2.2
        composited_linear = fg_linear * mask_3ch + bg_linear * (1.0 - mask_3ch)
        composited = np.clip(composited_linear, 0.0, 1.0) ** (1.0 / 2.2) * 255.0
    else:
        composited = fg_bgr * mask_3ch + bg_bgr * (1.0 - mask_3ch)

    return np.clip(composited, 0.0, 255.0).astype(np.uint8)
