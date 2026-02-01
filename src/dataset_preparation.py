from pathlib import Path
import shutil
import random

from src.label_budget import build_image_object_index, sample_by_object_budget, choose_boxes, mask_yolo_bboxes


def prepare_labeled_unlabeled_with_object_budget(
    images: list[Path],
    labels_dir: Path,
    labeled_img_dir: Path,
    labeled_lbl_dir: Path,
    unlabeled_img_dir: Path,
    target_objects: int,
    rng: random.Random,
    mask_mode: str = "mean",
):
    """
    Prepares:
    - labeled images with EXACT object budget (RGB-masked if needed)
    - unlabeled images = rest of images (original RGB, no labels)

    Returns:
        dict with statistics
    """

    labeled_img_dir.mkdir(parents=True, exist_ok=True)
    labeled_lbl_dir.mkdir(parents=True, exist_ok=True)
    unlabeled_img_dir.mkdir(parents=True, exist_ok=True)

    # --- build object index ---
    indexed = build_image_object_index(images, labels_dir)

    selected, total_objects = sample_by_object_budget(
        indexed,
        target_objects,
        rng,
    )

    used_images = set()
    remaining = target_objects
    actual_objects = 0

    # --- labeled part ---
    for img, lbl, n in selected:
        out_img = labeled_img_dir / img.name
        out_lbl = labeled_lbl_dir / lbl.name

        with open(lbl) as f:
            lines = f.readlines()

        keep = choose_boxes(len(lines), remaining, rng)

        mask_yolo_bboxes(
            img_path=img,
            label_path=lbl,
            out_img_path=out_img,
            out_label_path=out_lbl,
            keep_indices=keep,
            mask_mode=mask_mode,
        )

        used_images.add(img)
        actual_objects += len(keep)
        remaining -= len(keep)

        if remaining <= 0:
            break

    # --- unlabeled part ---
    for img in images:
        if img not in used_images:
            shutil.copy(img, unlabeled_img_dir / img.name)

    return {
        "n_labeled_images": len(used_images),
        "n_unlabeled_images": len(images) - len(used_images),
        "n_objects": actual_objects,
        "target_objects": target_objects,
    }
