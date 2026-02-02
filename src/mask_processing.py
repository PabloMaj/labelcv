import cv2
import numpy as np


def mask_to_polygon(mask: np.ndarray) -> np.ndarray | None:
    """
    Converts a binary mask (H x W) to a polygon (N x 2).
    Returns None if no valid polygon found.
    """
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)
    if len(cnt) < 3:
        return None

    return cnt.squeeze(1)


def polygon_to_bbox(poly: np.ndarray) -> tuple[int, int, int, int] | None:
    """
    Converts polygon Nx2 to bounding box (x1, y1, x2, y2).
    Returns None if polygon is invalid.
    """
    if poly is None or len(poly) == 0:
        return None
    xs = poly[:, 0]
    ys = poly[:, 1]
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
