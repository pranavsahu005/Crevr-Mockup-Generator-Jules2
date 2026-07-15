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
    feathered_mask: np.ndarray
) -> np.ndarray:
    """
    Performs alpha blending of foreground over background using a feathered mask.
    foreground: BGRA or BGR image (uint8, same dimensions as background).
    background: BGR image.
    feathered_mask: 2D float32 mask of shape (H, W) with values [0, 1].
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
    composited = fg_bgr * mask_3ch + bg_bgr * (1.0 - mask_3ch)

    return np.clip(composited, 0.0, 255.0).astype(np.uint8)

def remove_solid_background(image: np.ndarray, tolerance: int = 35) -> np.ndarray:
    """
    Automatically detects and removes solid backgrounds (chroma-keying)
    from uploaded design images (typically JPEGs with white or black backdrops).
    tolerance: distance threshold for color matching in BGR space.
    Returns BGRA image with transparent background.
    """
    h, w = image.shape[:2]
    # Ensure BGRA
    if image.shape[2] == 3:
        img_bgra = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    else:
        img_bgra = image.copy()

    # Sample BGR colors from the four extreme corners of the image
    corners = [
        img_bgra[0, 0, :3],
        img_bgra[0, w - 1, :3],
        img_bgra[h - 1, 0, :3],
        img_bgra[h - 1, w - 1, :3]
    ]

    # Find the median corner color to avoid outliers
    bg_color = np.median(corners, axis=0).astype(np.float32)

    # Calculate Euclidean distance of every pixel to the detected background color
    bgr_diff = img_bgra[:, :, :3].astype(np.float32) - bg_color
    distances = np.linalg.norm(bgr_diff, axis=2)

    # Pixels close to the background color are identified as background
    bg_mask = distances < tolerance

    # Set alpha channel to 0 for background pixels
    alpha = img_bgra[:, :, 3].copy()
    alpha[bg_mask] = 0

    # Smooth the alpha transition (feather edges of key) using a small blur
    alpha_float = alpha.astype(np.float32)
    alpha_smoothed = cv2.GaussianBlur(alpha_float, (3, 3), 0)

    img_bgra[:, :, 3] = np.clip(alpha_smoothed, 0, 255).astype(np.uint8)
    return img_bgra
