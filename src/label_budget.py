import cv2
import numpy as np
from pathlib import Path
import random


def count_objects(label_path: Path) -> int:
    if not label_path.exists():
        return 0
    with open(label_path) as f:
        return sum(1 for _ in f)


def build_image_object_index(images, labels_dir: Path):
    index = []
    for img in images:
        lbl = labels_dir / f"{img.stem}.txt"
        n = count_objects(lbl)
        if n > 0:
            index.append((img, lbl, n))
    return index


def sample_by_object_budget(indexed_images, target_objects, rng: random.Random):
    rng.shuffle(indexed_images)

    selected = []
    total = 0

    for img, lbl, n in indexed_images:
        if total >= target_objects:
            break
        selected.append((img, lbl, n))
        total += n

    return selected, total


def choose_boxes(n_boxes: int, max_objects: int, rng: random.Random):
    idx = list(range(n_boxes))
    if n_boxes <= max_objects:
        return idx
    return rng.sample(idx, max_objects)


def mask_yolo_bboxes(
    img_path: Path,
    label_path: Path,
    out_img_path: Path,
    out_label_path: Path,
    keep_indices: list[int],
    mask_mode: str = "mean",   # "mean" | "black"
    offset_px: int = 50,        # <<< NOWE
):
    """
    Masks all YOLO bboxes NOT in keep_indices.
    Bboxes are expanded by offset_px and clipped to image bounds.
    """

    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]

    with open(label_path) as f:
        lines = f.readlines()

    kept = [lines[i] for i in keep_indices]
    masked = [lines[i] for i in range(len(lines)) if i not in keep_indices]

    if mask_mode == "mean":
        fill_value = img.mean(axis=(0, 1)).astype(np.uint8)
    else:
        fill_value = 0

    for line in masked:
        _, cx, cy, bw, bh = map(float, line.split())

        # --- YOLO -> pixel coords ---
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)

        # --- apply offset ---
        x1 -= offset_px
        y1 -= offset_px
        x2 += offset_px
        y2 += offset_px

        # --- clip to image ---
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        if x2 > x1 and y2 > y1:
            img[y1:y2, x1:x2] = fill_value

    cv2.imwrite(str(out_img_path), img)

    with open(out_label_path, "w") as f:
        f.writelines(kept)
