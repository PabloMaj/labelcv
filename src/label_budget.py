import cv2
from pathlib import Path
import random
import numpy as np
import shutil


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


def mask_bboxes_or_polygons(img_path, label_path, out_img_path, out_label_path,
                            keep_indices, mask_mode="black", task="detect", bbox_offset_px: int = 0):
    # Read original image
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]

    # Read labels
    with open(label_path) as f:
        lines = [line.strip() for line in f.readlines()]

    # Keep only selected objects
    kept_lines = [lines[i] for i in keep_indices]
    with open(out_label_path, "w") as f:
        for line in kept_lines:
            f.write(line + "\n")

    # If no masking, just copy image
    if mask_mode is None:
        shutil.copy(img_path, out_img_path)
        return

    # Mask objects not kept
    for i, line in enumerate(lines):
        if i in keep_indices:
            continue

        parts = line.split()

        if task == "detect":
            # YOLO bbox: class x_center y_center width height (normalized)
            xc, yc, bw, bh = map(float, parts[1:])

            x1 = int((xc - bw / 2) * w)
            y1 = int((yc - bh / 2) * h)
            x2 = int((xc + bw / 2) * w)
            y2 = int((yc + bh / 2) * h)

            # --- APPLY OFFSET (padding) ---
            if bbox_offset_px > 0:
                x1 -= bbox_offset_px
                y1 -= bbox_offset_px
                x2 += bbox_offset_px
                y2 += bbox_offset_px

            # --- CLAMP TO IMAGE ---
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            if x2 <= x1 or y2 <= y1:
                continue  # safety guard

            if mask_mode == "mean":
                patch = img[y1:y2, x1:x2]
                if patch.size == 0:
                    continue
                color = patch.mean(axis=(0, 1)).astype(int)
                img[y1:y2, x1:x2] = color
            else:
                img[y1:y2, x1:x2] = 0  # black mask

        elif task == "segment":
            # YOLOv8 segmentation: class x1 y1 x2 y2 x3 y3 ...
            coords = list(map(float, parts[1:]))
            polygon = np.array([(int(coords[j] * w), int(coords[j + 1] * h)) for j in range(0, len(coords), 2)])
            if mask_mode == "mean":
                # approximate mean color of bounding rect
                x, y, bw, bh = cv2.boundingRect(polygon)
                patch = img[y:y + bh, x:x + bw]
                color = patch.mean(axis=(0, 1)).astype(int)
                cv2.fillPoly(img, [polygon], color.tolist())
            else:
                cv2.fillPoly(img, [polygon], (0, 0, 0))

    cv2.imwrite(str(out_img_path), img)
