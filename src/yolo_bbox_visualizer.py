import cv2
from pathlib import Path
from typing import Tuple


class YoloBBoxVisualizer:
    @staticmethod
    def load_yolo_labels(label_path: Path, img_shape: Tuple[int, int]):
        """
        Supports:
        - YOLO detection: class cx cy w h
        - YOLO instance segmentation: class x1 y1 x2 y2 ... xn yn
        Returns list of bounding boxes (x1, y1, x2, y2)
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

    @staticmethod
    def draw_red_boxes(image_path: Path, label_path: Path, out_path: Path):
        img = cv2.imread(str(image_path))
        h, w = img.shape[:2]

        boxes = YoloBBoxVisualizer.load_yolo_labels(label_path, (h, w))

        for x1, y1, x2, y2 in boxes:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        cv2.imwrite(str(out_path), img)
        return out_path
