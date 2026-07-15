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

def apply_corner_radius(img: np.ndarray, radius_px: float) -> np.ndarray:
    """
    Applies a rounded corner mask to the design image.
    radius_px: corner radius in pixels.
    Returns the rounded BGRA image.
    """
    if radius_px <= 0:
        return img
    h, w = img.shape[:2]
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

    mask = np.zeros((h, w), dtype=np.uint8)
    r = int(min(radius_px, min(h, w) // 2))
    if r <= 0:
        return img

    # Draw filled rectangles
    cv2.rectangle(mask, (r, 0), (w - r, h), 255, -1)
    cv2.rectangle(mask, (0, r), (w, h - r), 255, -1)

    # Draw corner circles
    cv2.circle(mask, (r, r), r, 255, -1)
    cv2.circle(mask, (w - r, r), r, 255, -1)
    cv2.circle(mask, (r, h - r), r, 255, -1)
    cv2.circle(mask, (w - r, h - r), r, 255, -1)

    # Soften the mask for anti-aliasing
    mask_float = mask.astype(np.float32)
    mask_smoothed = cv2.GaussianBlur(mask_float, (3, 3), 0)
    mask_norm = np.clip(mask_smoothed / 255.0, 0.0, 1.0)

    # Apply mask to alpha channel
    img[:, :, 3] = (img[:, :, 3].astype(np.float32) * mask_norm).astype(np.uint8)
    return img

def apply_perspective_bend(corners: list, bend_factor: float) -> list:
    """
    Dynamically adjusts the top width of the design's destination quad corners
    to narrow/widen the perspective of tilted screens (like angled laptops).
    bend_factor: percentage adjustment (e.g. -50 to 50).
    """
    if bend_factor == 0.0 or not corners or len(corners) != 4:
        return corners

    # corners: [tl, tr, br, bl]
    tl, tr, br, bl = np.array(corners, dtype=np.float32)

    # Calculate midpoint of top edge
    ct = (tl + tr) / 2.0

    # Compute vector from top center to corners
    v_tl = tl - ct
    v_tr = tr - ct

    # Shift top corners closer/farther
    scale_factor = 1.0 + (bend_factor / 100.0)
    new_tl = ct + v_tl * scale_factor
    new_tr = ct + v_tr * scale_factor

    return [new_tl.tolist(), new_tr.tolist(), br.tolist(), bl.tolist()]
