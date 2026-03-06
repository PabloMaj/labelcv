import os
import glob
import numpy as np
from tqdm import tqdm
from src.sam_model import SamSingleton
from src.grounding_dino_model import GroundingDinoSingleton
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics import YOLO
import tempfile
import json


# ---------------- Helper functions ----------------
def load_best_prompt(path_to_file):
    with open(path_to_file, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("BEST PROMPT") and not line.startswith("AP50"):
                return line


def yolo_to_bbox(label_line, img_width, img_height):
    parts = list(map(float, label_line.strip().split()))
    _, cx, cy, w, h = parts
    x1 = (cx - w / 2) * img_width
    y1 = (cy - h / 2) * img_height
    x2 = (cx + w / 2) * img_width
    y2 = (cy + h / 2) * img_height
    return [x1, y1, x2, y2]


def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    box2_area = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0


# ---------------- Dataset Loader ----------------
class DatasetLoader:
    def __init__(self, images_dir, labels_dir):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.image_paths = sorted(glob.glob(os.path.join(images_dir, "*.*")))

    def load_annotations(self, image_path):
        img_name = os.path.basename(image_path)
        label_file = os.path.join(self.labels_dir, os.path.splitext(img_name)[0] + ".txt")
        if not os.path.exists(label_file):
            return []
        img = Image.open(image_path)
        width, height = img.size
        with open(label_file, "r") as f:
            return [yolo_to_bbox(line, width, height) for line in f.readlines()]


# ---------------- Post Processor ----------------
class ResultsPostProcessor:
    def process(self, results=None, model_type: str = None):
        if model_type in ["sam", "yolo"]:
            boxes_coords = []
            conf_values = []
            boxes_all = results[0].boxes
            for box in boxes_all:
                boxes_coords.append(box.xyxy.cpu().numpy().tolist()[0])
                conf_values.append(box.conf.cpu().numpy().tolist()[0])
        elif model_type == "grounding_dino":
            boxes_coords = results[0]["boxes"].cpu().numpy().tolist()
            conf_values = results[0]["scores"].cpu().numpy().tolist()
        return boxes_coords, conf_values


# ---------------- Evaluator ----------------
class Evaluator:
    def __init__(self, model, model_type):
        self.model = model
        self.model_type = model_type
        self.post_processor = ResultsPostProcessor()

    def predict_once(self, image_path, prompt_filter="crack", base_conf=0.01):
        """Jednorazowe wnioskowanie z niskim thresholdem"""
        if self.model_type == "sam":
            results = self.model.predict(path_to_image=image_path, prompt_filter=prompt_filter)
        elif self.model_type == "grounding_dino":
            results = self.model.predict(path_to_image=image_path, prompt_filter=[[prompt_filter]], conf_thresh=base_conf)
        elif self.model_type == "yolo":
            results = self.model.predict(source=image_path, save=False, conf=0.01)
        boxes_coords, conf_values = self.post_processor.process(results, self.model_type)
        return boxes_coords, conf_values


# ---------------- AP50 Calculator ----------------
def compute_ap50(pred_boxes_all, gt_boxes_all):
    """
    Compute AP50 using pycocotools COCOeval.
    """

    images = []
    annotations = []
    detections = []

    ann_id = 1
    img_id_map = {}

    # ---- Build COCO GT ----
    for i, (img_path, gt_boxes) in enumerate(gt_boxes_all.items()):
        img_id = i + 1
        img_id_map[img_path] = img_id

        images.append({
            "id": img_id,
            "file_name": os.path.basename(img_path)
        })

        for gt in gt_boxes:
            x1, y1, x2, y2 = gt
            w = x2 - x1
            h = y2 - y1

            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": 1,
                "bbox": [x1, y1, w, h],
                "area": w * h,
                "iscrowd": 0
            })
            ann_id += 1

    coco_gt_dict = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "object"}]
    }

    # ---- Build detections ----
    for img_path, (boxes, confs) in pred_boxes_all.items():
        img_id = img_id_map[img_path]

        for box, score in zip(boxes, confs):
            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1

            detections.append({
                "image_id": img_id,
                "category_id": 1,
                "bbox": [x1, y1, w, h],
                "score": float(score)
            })

    # ---- Write temporary json ----
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f_gt:
        json.dump(coco_gt_dict, f_gt)
        gt_path = f_gt.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f_dt:
        json.dump(detections, f_dt)
        dt_path = f_dt.name

    # ---- Run COCOeval ----
    coco_gt = COCO(gt_path)
    coco_dt = coco_gt.loadRes(dt_path)

    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")

    coco_eval.params.iouThrs = np.array([0.5])  # AP50 only
    coco_eval.params.maxDets = [100, 300, 1000]

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    ap50 = coco_eval.stats[0]

    return float(ap50)


# ---------------- Threshold Optimizer ----------------
class ThresholdOptimizer:
    def __init__(self, precomputed_preds, dataset_loader):
        self.preds = precomputed_preds
        self.dataset_loader = dataset_loader

    def compute_metrics_for_thresh(self, conf_thresh):
        precisions, recalls, f1s = [], [], []
        for img_path in self.dataset_loader.image_paths:
            gt_boxes = self.dataset_loader.load_annotations(img_path)
            boxes_coords, conf_values = self.preds[img_path]
            pred_boxes = [b for b, c in zip(boxes_coords, conf_values) if c >= conf_thresh]
            matched_gt = set()
            matched_pred = set()
            for i, gt in enumerate(gt_boxes):
                for j, pred in enumerate(pred_boxes):
                    if j in matched_pred:
                        continue
                    if iou(gt, pred) >= 0.5:
                        matched_gt.add(i)
                        matched_pred.add(j)
                        break
            tp = len(matched_gt)
            fp = len(pred_boxes) - tp
            fn = len(gt_boxes) - tp
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
        return np.mean(precisions), np.mean(recalls), np.mean(f1s)

    def optimize_threshold(self, start=0.05, end=1.0, step=0.01):
        best_f1 = 0
        best_thresh = start
        thresh_results = []
        for conf_thresh in np.arange(start, end, step):
            p, r, f1 = self.compute_metrics_for_thresh(conf_thresh)
            thresh_results.append((conf_thresh, f1))
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = conf_thresh
        return best_thresh, best_f1, thresh_results


# ---------------- Full Evaluator ----------------
class FullEvaluator:
    def __init__(self, precomputed_preds, dataset_loader):
        self.preds = precomputed_preds
        self.dataset_loader = dataset_loader

    def evaluate(self):
        # Prepare GT and pred boxes
        gt_boxes_all = {img: self.dataset_loader.load_annotations(img) for img in self.dataset_loader.image_paths}
        pred_boxes_all = {}
        for img_path in tqdm(self.dataset_loader.image_paths):
            boxes, confs = self.preds[img_path]
            pred_boxes_all[img_path] = (boxes, confs)

        # Compute AP50
        ap50 = compute_ap50(pred_boxes_all, gt_boxes_all)

        return {"AP50Box": ap50}


# =======================
# MAIN LOOP - MULTIPLE DATASETS
# =======================
if __name__ == "__main__":
    DATASET_NAMES = ["bee_subset_1", "african-wildlife_buffalo", "african-wildlife_elephant", "african-wildlife_rhino",
                     "african-wildlife_zebra", "construction-ppe_vest", "wheat_arvalis_1", "wheat_arvalis_3",
                     "wheat_ethz_1", "wheat_rres_1", "homeobjects_chair", "homeobjects_table", "homeobjects_potted_plant",
                     "homeobjects_lamp", "crack-seg"][:-1]
    obj_list = [80]
    repeat_seed_list = [(0, 42), (1, 43), (2, 44)]

    base_data_dir = "C:/projects/labelcv/data/datasets"
    base_runs_dir = "C:/projects/labelcv/runs"

    model_path_sam = "resources/sam3.pt"
    save_flag = False

    model_sam = SamSingleton(model_path=model_path_sam, conf_thresh=0.01, save_flag=save_flag)
    model_grounding_dino = GroundingDinoSingleton()
    model_map = {"sam": model_sam, "grounding_dino": model_grounding_dino}

    for method_type in ["yolo", "sam", "grounding_dino"][1:]:
        for dataset_name in DATASET_NAMES:
            print(f"\n=== DATASET: {dataset_name} ===")
            val_images = os.path.join(base_data_dir, dataset_name, "images", "val")
            val_labels = os.path.join(base_data_dir, dataset_name, "labels", "val")
            test_images = os.path.join(base_data_dir, dataset_name, "images", "test")
            test_labels = os.path.join(base_data_dir, dataset_name, "labels", "test")

            for obj_n in obj_list:
                for repeat_x, seed_y in repeat_seed_list:
                    prompt_path = os.path.join(base_runs_dir, dataset_name, "sam", f"obj_{obj_n}", f"repeat_{repeat_x}_seed_{seed_y}", "best_prompt.txt")
                    prompt_filter = load_best_prompt(prompt_path)
                    print(f"\nDataset={dataset_name}, obj={obj_n}, repeat={repeat_x}_seed={seed_y}, prompt='{prompt_filter}'")

                    if method_type == "yolo":
                        path_to_weights = os.path.join(base_runs_dir, dataset_name, "sam", f"obj_{obj_n}", f"repeat_{repeat_x}_seed_{seed_y}",
                                                       "yolo_train", "weights", "best.pt")
                        model_map[method_type] = YOLO(path_to_weights)

                    evaluator = Evaluator(model_map[method_type], method_type)
                    val_dataset = DatasetLoader(val_images, val_labels)
                    test_dataset = DatasetLoader(test_images, test_labels)

                    """
                    precomputed_val = {}
                    for img_path in tqdm(val_dataset.image_paths):
                        boxes_coords, conf_values = evaluator.predict_once(img_path, prompt_filter, base_conf=0.01)
                        precomputed_val[img_path] = (boxes_coords, conf_values)

                    optimizer = ThresholdOptimizer(precomputed_val, val_dataset)
                    best_thresh, best_f1, thresh_results = optimizer.optimize_threshold()
                    print(f"Optimal threshold: {best_thresh:.2f}, F1: {best_f1:.4f}")
                    """

                    print(f"prompt_filter={prompt_filter}")
                    precomputed_test = {img: evaluator.predict_once(img, prompt_filter, base_conf=0.01) for img in tqdm(test_dataset.image_paths)}

                    full_eval = FullEvaluator(precomputed_test, test_dataset)
                    test_metrics = full_eval.evaluate()

                    path_to_save_results = f"C:/projects/labelcv/runs/{dataset_name}/vanilla_{method_type}/obj_{obj_n}/repeat_{repeat_x}_seed_{seed_y}/eval_test/"
                    os.makedirs(path_to_save_results, exist_ok=True)
                    file_path = os.path.join(path_to_save_results, "metrics.json")
                    with open(file_path, "w", encoding="utf-8") as f:
                        json.dump(test_metrics, f, ensure_ascii=False, indent=4)
