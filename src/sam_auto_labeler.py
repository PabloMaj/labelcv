from pathlib import Path
import cv2
from typing import Optional
from tqdm import tqdm

from src.sam_model import SamSingleton
from src.yolo_bbox_visualizer import YoloBBoxVisualizer
from src.utils import mask_to_bbox, masks_to_image_space


class SAM3AutoLabeler:
    def __init__(
        self,
        sam: SamSingleton,
        class_id: int = 0,
        img_exts=(".jpg", ".jpeg", ".png"),
    ):
        """
        sam       : instancja SamSingleton
        class_id : ID klasy w YOLO
        """
        self.sam = sam
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
        image_dir = Path(image_dir)
        out_label_dir = Path(out_label_dir)
        out_label_dir.mkdir(parents=True, exist_ok=True)

        if visualize:
            assert vis_dir is not None, "vis_dir must be provided if visualize=True"
            vis_dir = Path(vis_dir)
            vis_dir.mkdir(parents=True, exist_ok=True)

        images = []
        for ext in self.img_exts:
            images.extend(image_dir.glob(f"*{ext}"))
        images = sorted(images)

        for img_path in tqdm(images, desc="Auto-labeling with SAM3"):
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
        bboxes = []

        for mask in masks:
            bbox = mask_to_bbox(mask)
            if bbox is None:
                continue
            bboxes.append(bbox)

        label_path = out_label_dir / f"{img_path.stem}.txt"
        self._save_yolo_labels(label_path, bboxes, w, h)

        if visualize:
            out_img_path = vis_dir / img_path.name
            YoloBBoxVisualizer.draw_red_boxes(
                image_path=img_path,
                label_path=label_path,
                out_path=out_img_path,
            )

    def _save_yolo_labels(
        self,
        label_path: Path,
        bboxes,
        img_w: int,
        img_h: int,
    ):
        lines = []

        for x1, y1, x2, y2 in bboxes:
            cx = ((x1 + x2) / 2) / img_w
            cy = ((y1 + y2) / 2) / img_h
            bw = (x2 - x1) / img_w
            bh = (y2 - y1) / img_h

            lines.append(
                f"{self.class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
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
