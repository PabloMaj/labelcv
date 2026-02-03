from pathlib import Path
import cv2
from typing import Optional
from tqdm import tqdm

from src.sam_model import SamSingleton
from src.yolo_bbox_visualizer import YoloBBoxVisualizer, YoloSegVisualizer
from src.mask_processing import mask_to_polygon, polygon_to_bbox, masks_to_image_space


class SAM3AutoLabeler:
    def __init__(
        self,
        sam: SamSingleton,
        task: str = "detect",   # "detect" | "segment"
        class_id: int = 0,
        img_exts=(".jpg", ".jpeg", ".png"),
    ):
        assert task in {"detect", "segment"}
        self.sam = sam
        self.task = task
        self.class_id = class_id
        self.img_exts = img_exts

    def label_folder(
        self,
        image_dir: Path,
        out_label_dir: Path,
        prompt_filter: Optional[str] = None,
        visualize: bool = False,
        vis_dir: Optional[Path] = None,
    ):
        out_label_dir.mkdir(parents=True, exist_ok=True)

        if visualize:
            assert vis_dir is not None
            vis_dir.mkdir(parents=True, exist_ok=True)

        images = []
        for ext in self.img_exts:
            images.extend(image_dir.glob(f"*{ext}"))
        images = sorted(images)

        for img_path in tqdm(images, desc=f"SAM3 auto-labeling [{self.task}]"):
            self._process_single_image(
                img_path,
                out_label_dir,
                prompt_filter,
                visualize,
                vis_dir,
            )

    def _process_single_image(
        self,
        img_path: Path,
        out_label_dir: Path,
        prompt_filter: Optional[str],
        visualize: bool,
        vis_dir: Optional[Path],
    ):
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]

        results = self.sam.predict(
            path_to_image=str(img_path),
            prompt_filter=prompt_filter,
        )

        masks = masks_to_image_space(results[0].masks)

        label_path = out_label_dir / f"{img_path.stem}.txt"

        if self.task == "detect":
            bboxes = self._masks_to_bboxes(masks)
            self._save_yolo_det_labels(label_path, bboxes, w, h)

        else:  # segment
            polygons = self._masks_to_polygons(masks)
            self._save_yolo_seg_labels(label_path, polygons, w, h)

        if visualize:
            out_img_path = vis_dir / img_path.name
            if self.task == "detect":
                YoloBBoxVisualizer.draw_red_boxes(
                    img_path, label_path, out_img_path
                )
            else:
                YoloSegVisualizer.draw_polygons(
                    img_path, label_path, out_img_path
                )

    # ======================================================
    # MASK → STRUCTURE
    # ======================================================

    def _masks_to_polygons(self, masks):
        polys = []
        for mask in masks:
            poly = mask_to_polygon(mask)
            if poly is not None:
                polys.append(poly)
        return polys

    def _masks_to_bboxes(self, masks):
        bboxes = []
        for mask in masks:
            poly = mask_to_polygon(mask)
            if poly is None:
                continue
            bboxes.append(polygon_to_bbox(poly))
        return bboxes

    # ======================================================
    # SAVE LABELS
    # ======================================================

    def _save_yolo_det_labels(self, label_path, bboxes, w, h):
        lines = []
        for x1, y1, x2, y2 in bboxes:
            cx = ((x1 + x2) / 2) / w
            cy = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            lines.append(
                f"{self.class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
            )

        with open(label_path, "w") as f:
            f.write("\n".join(lines))

    def _save_yolo_seg_labels(self, label_path, polygons, w, h):
        lines = []
        for poly in polygons:
            coords = []
            for x, y in poly:
                coords.append(f"{x / w:.6f}")
                coords.append(f"{y / h:.6f}")
            lines.append(
                f"{self.class_id} " + " ".join(coords)
            )

        with open(label_path, "w") as f:
            f.write("\n".join(lines))


if __name__ == "__main__":

    prompt_filter = "Pale yellow, rounded tablets with a speckled, textured surface"

    model_sam = SamSingleton(
        model_path="resources/sam3.pt",
        conf_thresh=0.25
    )

    labeler = SAM3AutoLabeler(model_sam, class_id=0)

    labeler.label_folder(
        image_dir=Path("runs/sam_pool_labelling/images"),
        out_label_dir=Path("runs/sam_pool_labelling/labels"),
        prompt_filter=prompt_filter,
        visualize=True,
        vis_dir=Path("runs/sam_pool_labelling/vis")
    )
