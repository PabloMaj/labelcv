import json
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict

from scipy.optimize import linear_sum_assignment
from skimage.draw import polygon as sk_polygon

from sam_model import SamSingleton
from ultralytics.engine.results import Masks


class InstanceSegmentationMAP:
    def __init__(self, iou_thresholds=None):
        self.iou_thresholds = (
            iou_thresholds
            if iou_thresholds is not None
            else np.arange(0.5, 1.0, 0.05)
        )

    # ==========================================================
    # GT (LabelMe)
    # ==========================================================

    @staticmethod
    def load_labelme_gt(json_path: Path) -> tuple[list[np.ndarray], tuple[int, int]]:
        with open(json_path, "r") as f:
            data = json.load(f)

        h, w = data["imageHeight"], data["imageWidth"]
        masks = []

        for shape in data["shapes"]:
            if shape["shape_type"] != "polygon":
                continue

            poly = np.array(shape["points"], dtype=np.float32)
            rr, cc = sk_polygon(poly[:, 1], poly[:, 0], (h, w))

            m = np.zeros((h, w), dtype=bool)
            m[rr, cc] = True
            masks.append(m)

        return masks, (h, w)

    # ==========================================================
    # Predykcje (Ultralytics Masks)
    # ==========================================================

    @staticmethod
    def masks_to_image_space(masks: Masks) -> List[np.ndarray]:
        """
        Masks -> list[bool ndarray] w przestrzeni ORYGINALNEGO obrazu
        """
        if masks is None or masks.data is None:
            return []

        h0, w0 = masks.orig_shape
        out = []

        for m in masks.data:
            m_np = m.cpu().numpy().astype(np.uint8)

            if m_np.shape != (h0, w0):
                m_np = cv2.resize(
                    m_np,
                    (w0, h0),
                    interpolation=cv2.INTER_NEAREST,
                )

            out.append(m_np.astype(bool))

        return out

    # ==========================================================
    # IoU + matching
    # ==========================================================

    @staticmethod
    def compute_iou(a: np.ndarray, b: np.ndarray) -> float:
        inter = np.logical_and(a, b).sum()
        union = np.logical_or(a, b).sum()
        return inter / union if union > 0 else 0.0

    def iou_matrix(self, gt_masks, pred_masks) -> np.ndarray:
        mat = np.zeros((len(gt_masks), len(pred_masks)), dtype=np.float32)

        for i, g in enumerate(gt_masks):
            for j, p in enumerate(pred_masks):
                mat[i, j] = self.compute_iou(g, p)

        return mat

    # ==========================================================
    # AP
    # ==========================================================

    def ap_at_iou(self, iou_matrix: np.ndarray, thr: float) -> float:
        if iou_matrix.size == 0:
            return 0.0

        cost = 1.0 - iou_matrix
        r, c = linear_sum_assignment(cost)

        tp = (iou_matrix[r, c] >= thr).sum()
        fp = iou_matrix.shape[1] - tp
        fn = iou_matrix.shape[0] - tp

        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0

        return p * r

    # ==========================================================
    # Public API
    # ==========================================================

    def evaluate(
        self,
        gt_json_paths: List[Path],
        pred_masks_list: List[Masks],
    ) -> Dict[str, float]:

        assert len(gt_json_paths) == len(pred_masks_list)

        ap_per_thr = {t: [] for t in self.iou_thresholds}

        for gt_path, pred_masks in zip(gt_json_paths, pred_masks_list):
            gt_masks, _ = self.load_labelme_gt(gt_path)
            pred = self.masks_to_image_space(pred_masks)

            iou_mat = self.iou_matrix(gt_masks, pred)

            for thr in self.iou_thresholds:
                ap = self.ap_at_iou(iou_mat, thr)
                ap_per_thr[thr].append(ap)

        results = {
            f"AP@{t:.2f}": float(np.mean(v))
            for t, v in ap_per_thr.items()
        }
        results["mAP"] = float(np.mean(list(results.values())))
        return results


if __name__ == "__main__":

    # settings
    model_path = "resources/sam3.pt"
    conf_thresh = 0.25
    save_flag = False
    prompt_filter = "bee"

    root_path_to_images = "data/metric_calculator/gts/"
    img_names = ["bee_img_1.jpg", "bee_img_2.jpg", "bee_img_3.jpg", "bee_img_4.jpg"]
    paths_to_images = [root_path_to_images + img_name for img_name in img_names]

    model_sam = SamSingleton(model_path=model_path, conf_thresh=conf_thresh, save_flag=save_flag)

    # predictions
    predictions_sam = []
    for path_to_img in paths_to_images:
        results = model_sam.predict(path_to_image=path_to_img, prompt_filter=prompt_filter)
        predictions_sam.append(results[0].masks)

    # ground truths
    gt_json_paths = [path_.replace("jpg", "json") for path_ in paths_to_images]

    # seg metric calculator
    seg_metric_calculator = InstanceSegmentationMAP()
    metric_seg = seg_metric_calculator.evaluate(gt_json_paths=gt_json_paths, pred_masks_list=predictions_sam)
    print(f"metric_seg={metric_seg}")
