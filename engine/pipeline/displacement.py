import cv2
import numpy as np

def apply_displacement_map(
    image: np.ndarray,
    displacement_map: np.ndarray,
    fold_intensity: float
) -> np.ndarray:
    """
    Applies local pixel offsets to an image using a grayscale displacement map.
    The map is converted into horizontal and vertical offsets (gradients) and applied via cv2.remap.
    image: BGRA (or BGR) numpy array.
    displacement_map: Grayscale 2D array, same spatial dimensions as the target image,
                      where 128 is neutral height, < 128 is a fold indentation, > 128 is peak.
    fold_intensity: multiplier factor for displacement scaling (usually 5 to 30 px).
    """
    h, w = image.shape[:2]

    # Check that displacement map matches image dimensions
    if displacement_map.shape[:2] != (h, w):
        displacement_map = cv2.resize(displacement_map, (w, h), interpolation=cv2.INTER_LINEAR)

    # Calculate gradients of displacement map as offsets
    # Grayscale map encodes relative height. Differences in height (slope) cause optical displacement.
    # We use Sobel to estimate the slope in x and y.
    sobel_x = cv2.Sobel(displacement_map, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(displacement_map, cv2.CV_32F, 0, 1, ksize=3)

    # Create coordinate grid
    y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)

    # Calculate the per-pixel displacement offsets
    # Normalize displacement factors.
    # We scale Sobel gradients so that larger folds map to a proportional pixel shift.
    dx = sobel_x * (fold_intensity / 128.0)
    dy = sobel_y * (fold_intensity / 128.0)

    map_x = x_coords + dx
    map_y = y_coords + dy

    # Apply displacement via cv2.remap with linear interpolation
    displaced_image = cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0)
    )

    return displaced_image
