from pathlib import Path
import cv2
from typing import Optional, Iterable
from tqdm import tqdm

from src.yolo_bbox_visualizer import YoloBBoxVisualizer

from typing import List, Tuple
from ultralytics import YOLO


class WeakYoloSingleton:
    _instance = None

    def __new__(cls, model_path: str, conf_thresh: float = 0.25, iou_thresh: float = 0.7, device: str = "cuda"):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_model(model_path, conf_thresh, iou_thresh, device)
        return cls._instance

    def _init_model(self, model_path: str, conf_thresh: float, iou_thresh: float, device: str):
        self.model_path = model_path
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.device = device
        self.model = YOLO(model_path)
        self.model.to(device)

    def predict(self, path_to_image: str) -> List[Tuple[int, int, int, int, float, int, Optional[List[Tuple[float, float]]]]]:
        results = self.model.predict(
            source=path_to_image,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            device=self.device,
            verbose=False,
        )

        detections = []

        if len(results) == 0:
            return detections

        r = results[0]
        boxes = r.boxes
        masks = r.masks

        if boxes is None:
            return detections

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)

        if masks is not None:
            polys = masks.xy
        else:
            polys = [None] * len(xyxy)

        for (x1, y1, x2, y2), conf, cls_id, poly in zip(xyxy, confs, clss, polys):
            detections.append((int(x1), int(y1), int(x2), int(y2), float(conf), int(cls_id), poly))

        return detections


class YOLOWeakAutoLabeler:
    def __init__(self, yolo: WeakYoloSingleton, class_id: int = 0, conf_thresh: float = 0.25, task: str = "detect", img_exts=(".jpg", ".jpeg", ".png")):
        self.yolo = yolo
        self.class_id = class_id
        self.conf_thresh = conf_thresh
        self.task = task
        self.img_exts = img_exts

    def label_folder(self, image_dir: Path, out_label_dir: Path, visualize: bool = False, vis_dir: Optional[Path] = None):
        image_dir = Path(image_dir)
        out_label_dir = Path(out_label_dir)
        out_label_dir.mkdir(parents=True, exist_ok=True)

        if visualize:
            assert vis_dir is not None
            vis_dir = Path(vis_dir)
            vis_dir.mkdir(parents=True, exist_ok=True)

        images = sorted([p for ext in self.img_exts for p in image_dir.glob(f"*{ext}")])

        for img_path in tqdm(images, desc="Auto-labeling with weak YOLO"):
            self._process_single_image(img_path, out_label_dir, visualize, vis_dir)

    def _process_single_image(self, img_path: Path, out_label_dir: Path, visualize: bool, vis_dir: Optional[Path]):
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]

        detections = self.yolo.predict(str(img_path))

        bboxes = []
        segments = []

        for det in detections:
            x1, y1, x2, y2, conf, cls, poly = det
            if conf < self.conf_thresh:
                continue

            if self.task == "detect":
                bboxes.append((x1, y1, x2, y2))
            elif self.task == "segment" and poly is not None:
                segments.append(poly)

        label_path = out_label_dir / f"{img_path.stem}.txt"

        if self.task == "detect":
            self._save_yolo_labels(label_path, bboxes, w, h)
        else:
            self._save_yolo_seg_labels(label_path, segments, w, h)

        if visualize:
            out_img_path = vis_dir / img_path.name
            YoloBBoxVisualizer.draw_red_boxes(image_path=img_path, label_path=label_path, out_path=out_img_path)

    def _save_yolo_labels(self, label_path: Path, bboxes: Iterable, img_w: int, img_h: int):
        lines = []
        for x1, y1, x2, y2 in bboxes:
            cx = ((x1 + x2) / 2) / img_w
            cy = ((y1 + y2) / 2) / img_h
            bw = (x2 - x1) / img_w
            bh = (y2 - y1) / img_h
            lines.append(f"{self.class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        with open(label_path, "w") as f:
            f.write("\n".join(lines))

    def _save_yolo_seg_labels(self, label_path: Path, segments: Iterable, img_w: int, img_h: int):
        lines = []
        for poly in segments:
            poly_norm = [f"{px / img_w:.6f} {py / img_h:.6f}" for px, py in poly]
            lines.append(f"{self.class_id} " + " ".join(poly_norm))
        with open(label_path, "w") as f:
            f.write("\n".join(lines))


if __name__ == "__main__":

    model_yolo = WeakYoloSingleton(
        model_path="runs/detect/train/weights/best.pt",
        conf_thresh=0.25
    )

    labeler = YOLOWeakAutoLabeler(
        yolo=model_yolo,
        class_id=0,
        conf_thresh=0.3
    )

    labeler.label_folder(
        image_dir=Path("runs/weak_yolo_labelling/images"),
        out_label_dir=Path("runs/weak_yolo_labelling/labels"),
        visualize=True,
        vis_dir=Path("runs/weak_yolo_labelling/vis"),
    )
