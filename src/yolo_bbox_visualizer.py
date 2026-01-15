import cv2
from pathlib import Path
from typing import Tuple


class YoloBBoxVisualizer:
    @staticmethod
    def load_yolo_labels(label_path: Path, img_shape: Tuple[int, int]):
        """
        YOLO format: class cx cy w h (normalized)
        """
        h, w = img_shape
        boxes = []

        with open(label_path) as f:
            for line in f:
                _, cx, cy, bw, bh = map(float, line.split())
                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)
                boxes.append((x1, y1, x2, y2))

        return boxes

    @staticmethod
    def draw_red_boxes(image_path: Path, label_path: Path, out_path: Path):
        img = cv2.imread(str(image_path))
        h, w = img.shape[:2]

        boxes = YoloBBoxVisualizer.load_yolo_labels(label_path, (h, w))

        for x1, y1, x2, y2 in boxes:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        cv2.imwrite(str(out_path), img)
        return out_path
