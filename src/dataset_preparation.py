from pathlib import Path
import shutil
import random
from typing import List

from src.label_budget import build_image_object_index, sample_by_object_budget, choose_boxes, mask_bboxes_or_polygons

from pathlib import Path
import numpy as np
import cv2 as cv

def imread(filename: str, flags: int = cv.IMREAD_COLOR) -> np.ndarray | None:
    try:
        file_bytes = np.fromfile(filename, dtype=np.uint8)
        if file_bytes.size == 0:
            return None

        im = cv.imdecode(file_bytes, flags)
        return im[..., None] if im is not None and im.ndim == 2 else im

    except (FileNotFoundError, OSError, ValueError):
        return None


def cache_image_to_disk(f: Path) -> None:
    str_f = f.as_posix()
    npy_path = str_f.rsplit(".", 1)[0] + ".npy"

    if not Path(npy_path).exists():
        img = imread(str_f)
        if img is not None:
            np.save(npy_path, img, allow_pickle=False)


def prepare_labeled_unlabeled_with_object_budget(
    images: List[Path],
    labels_dir: Path,
    labeled_img_dir: Path,
    labeled_lbl_dir: Path,
    unlabeled_img_dir: Path,
    target_objects: int,
    rng: random.Random,
    mask_mode: str = "mean",
    task: str = "detect",  # "detect" | "segment"
):
    labeled_img_dir.mkdir(parents=True, exist_ok=True)
    labeled_lbl_dir.mkdir(parents=True, exist_ok=True)
    unlabeled_img_dir.mkdir(parents=True, exist_ok=True)

    indexed = build_image_object_index(images, labels_dir)

    selected, total_objects = sample_by_object_budget(indexed, target_objects, rng)

    used_images = set()
    remaining = target_objects
    actual_objects = 0

    for img, lbl, n in selected:
        out_img = labeled_img_dir / img.name
        out_lbl = labeled_lbl_dir / lbl.name

        with open(lbl) as f:
            lines = f.readlines()

        keep = choose_boxes(len(lines), remaining, rng)

        mask_bboxes_or_polygons(
            img_path=img,
            label_path=lbl,
            out_img_path=out_img,
            out_label_path=out_lbl,
            keep_indices=keep,
            mask_mode=mask_mode,
            task=task,
        )

        used_images.add(img)
        actual_objects += len(keep)
        remaining -= len(keep)
        if remaining <= 0:
            break

    for img in images:
        if img not in used_images:
            shutil.copy(img, unlabeled_img_dir / img.name)
            path = unlabeled_img_dir / img.name
        else:
            path = labeled_img_dir / img.name

        # Cache the image to disk as a .npy file for faster loading during training
        cache_image_to_disk(path)

    return {
        "n_labeled_images": len(used_images),
        "n_unlabeled_images": len(images) - len(used_images),
        "n_objects": actual_objects,
        "target_objects": target_objects,
    }
