import cv2
from pathlib import Path
import random
from typing import List
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


def mask_yolo_bboxes(
    img_path: Path,
    label_path: Path,
    out_img_path: Path,
    out_label_path: Path,
    keep_indices: List[int],
    mask_mode: str = "mean",
    task: str = "detect",
):
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]

    with open(label_path) as f:
        lines = [line.strip() for line in f.readlines()]

    new_lines = []

    for i in keep_indices:
        line = lines[i]
        if not line:
            continue

        parts = line.split()
        if len(parts) < 5:
            continue

        cls_id = parts[0]
        cx, cy, bw, bh = map(float, parts[1:5])

        if task == "segment" and len(parts) > 5:
            seg_points = list(map(float, parts[5:]))
        else:
            seg_points = []

        if mask_mode == "mean":
            out_img = img.copy()
            # opcjonalnie można dodać maskowanie obiektu
            # np. wypełnić bbox / polygon średnią kolorem tła
            # tutaj zostawiamy bez zmiany
            cv2.imwrite(str(out_img_path), out_img)
        else:
            shutil.copy(img_path, out_img_path)

        # zapis nowej linii w tym samym formacie
        line_out = f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
        if seg_points:
            line_out += " " + " ".join(f"{p:.6f}" for p in seg_points)
        new_lines.append(line_out)

    with open(out_label_path, "w") as f:
        f.write("\n".join(new_lines))
