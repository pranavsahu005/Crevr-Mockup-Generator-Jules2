import cv2
import numpy as np

def compute_perspective_matrix(src_pts: list, dst_pts: list) -> np.ndarray:
    """
    Computes the 3x3 perspective homography matrix.
    src_pts and dst_pts should be lists/arrays of 4 points: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]].
    """
    src = np.array(src_pts, dtype=np.float32)
    dst = np.array(dst_pts, dtype=np.float32)
    return cv2.getPerspectiveTransform(src, dst)

def warp_design_to_surface(design: np.ndarray, matrix: np.ndarray, target_size: tuple) -> np.ndarray:
    """
    Warps the design image using the 3x3 homography matrix into the target dimensions.
    target_size is (width, height).
    Uses linear interpolation by default.
    Returns warped image with transparency channel (BGRA).
    """
    # Ensure design is BGRA to maintain alpha transparency
    if design.shape[2] == 3:
        design = cv2.cvtColor(design, cv2.COLOR_BGR2BGRA)

    warped = cv2.warpPerspective(
        design,
        matrix,
        target_size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0)
    )
    return warped
