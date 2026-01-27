import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List
from scipy.optimize import linear_sum_assignment

from ultralytics.engine.results import Masks


class DetectionAP50:
    def evaluate_image(self, gt_boxes, pred_boxes):
        if len(gt_boxes) == 0:
            return 0.0

        if len(pred_boxes) == 0:
            return 0.0

        iou_mat = np.zeros((len(gt_boxes), len(pred_boxes)))

        for i, g in enumerate(gt_boxes):
            for j, p in enumerate(pred_boxes):
                iou_mat[i, j] = bbox_iou(g, p)

        cost = 1.0 - iou_mat
        r, c = linear_sum_assignment(cost)

        tp = (iou_mat[r, c] >= 0.5).sum()
        fp = len(pred_boxes) - tp
        fn = len(gt_boxes) - tp

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0

        return precision * recall  # AP@0.50 (uprośćmy)

    def evaluate_dataset(self, gt_list, pred_list):
        scores = [
            self.evaluate_image(gt, pr)
            for gt, pr in zip(gt_list, pred_list)
        ]
        return float(np.mean(scores))


def mask_to_bbox(mask: np.ndarray) -> Tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return xs.min(), ys.min(), xs.max(), ys.max()


def load_yolo_bboxes(label_path, img_shape):
    """
    Supports:
    - YOLO detection: class cx cy w h
    - YOLO instance segmentation: class x1 y1 x2 y2 ... xn yn
    Returns list of (x1, y1, x2, y2)
    """
    h, w = img_shape
    boxes = []

    with open(label_path) as f:
        for line in f:
            parts = list(map(float, line.split()))
            coords = parts[1:]

            # --- klasyczny YOLO bbox ---
            if len(coords) == 4:
                cx, cy, bw, bh = coords
                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)
                boxes.append((x1, y1, x2, y2))

            # --- segmentacja instancyjna → bbox z punktów ---
            elif len(coords) >= 6 and len(coords) % 2 == 0:
                xs = []
                ys = []
                for i in range(0, len(coords), 2):
                    xs.append(int(coords[i] * w))
                    ys.append(int(coords[i + 1] * h))

                x1, y1 = min(xs), min(ys)
                x2, y2 = max(xs), max(ys)
                boxes.append((x1, y1, x2, y2))

    return boxes


def bbox_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)

    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def masks_to_image_space(masks: Masks) -> List[np.ndarray]:
    """
    Konwersja masek SAM3 do listy masek bool
    w przestrzeni oryginalnego obrazu (H, W)
    """
    if masks is None or masks.data is None:
        return []

    h0, w0 = masks.orig_shape
    out = []

    for m in masks.data:
        m_np = m.cpu().numpy().astype(np.uint8)

        if m_np.shape != (h0, w0):
            m_np = cv2.resize(
                m_np,
                (w0, h0),
                interpolation=cv2.INTER_NEAREST,
            )

        out.append(m_np.astype(bool))

    return out


def load_images_and_labels(
    image_dir: Path,
    label_dir: Path,
    img_exts=(".png", ".jpg", ".jpeg"),
):
    images = []
    for ext in img_exts:
        images.extend(image_dir.glob(f"*{ext}"))

    images = sorted(images)
    labels = [label_dir / f"{p.stem}.txt" for p in images]

    assert all(label.exists() for label in labels), "Missing label files"
    return images, labels
