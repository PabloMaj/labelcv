
import cv2
import re
import json
import spacy
from pathlib import Path

from src.prompt_generator import PromptGenerator
from src.llm_model import OllamaVisionLLM
from src.sam_model import SamSingleton
from src.yolo_bbox_visualizer import YoloBBoxVisualizer, YoloSegVisualizer
from src.metrics import DetectionAP50, SegmentationAP50
from src.loaders import load_yolo_bboxes, load_yolo_segment_masks, load_images_and_labels
from src.mask_processing import mask_to_bbox, masks_to_image_space

nlp = spacy.load("en_core_web_sm")


class SAM3PromptOptimizer:
    def __init__(
        self,
        sam_model_path: str,
        llm_model: str,
        task: str,  # "detect" | "segment"
        max_iters: int = 5,
        max_desc_words: int = 10,
    ):
        assert task in ("detect", "segment")

        self.task = task
        self.sam = SamSingleton(model_path=sam_model_path)
        self.llm = OllamaVisionLLM(llm_model)

        self.metric = (
            DetectionAP50()
            if task == "detect"
            else SegmentationAP50()
        )

        self.max_iters = max_iters
        self.max_desc_words = max_desc_words

        self.history_all = []
        self.history_topk = []

    # --------------------------------------------------
    # Visualization
    # --------------------------------------------------
    def _visualize_annotations(self, img, lbl, out_path):
        if self.task == "detect":
            YoloBBoxVisualizer.draw_red_boxes(img, lbl, out_path)
        else:
            YoloSegVisualizer.draw_polygons(img, lbl, out_path)

    # --------------------------------------------------
    # SAM inference
    # --------------------------------------------------
    def _predict_masks(self, images, prompt):
        preds = []
        for img in images:
            results = self.sam.predict(
                path_to_image=str(img),
                prompt_filter=prompt
            )
            preds.append(results[0].masks)
        return preds

    # --------------------------------------------------
    # Evaluation
    # --------------------------------------------------
    def _evaluate(self, val_images, val_labels, pred_masks_all):
        if self.task == "detect":
            gt_boxes_all = []
            pred_boxes_all = []

            for img, lbl, masks in zip(val_images, val_labels, pred_masks_all):
                img_np = cv2.imread(str(img))
                h, w = img_np.shape[:2]

                gt_boxes = load_yolo_bboxes(lbl, (h, w))

                pred_boxes = []
                for m in masks_to_image_space(masks):
                    bb = mask_to_bbox(m)
                    if bb is not None:
                        pred_boxes.append(bb)

                gt_boxes_all.append(gt_boxes)
                pred_boxes_all.append(pred_boxes)

            return self.metric.evaluate_dataset(
                gt_boxes_all,
                pred_boxes_all
            )

        else:
            gt_masks_all = []
            pred_masks_all_out = []

            for masks in pred_masks_all:
                pred_masks_all_out.append(masks_to_image_space(masks))

            for img, lbl in zip(val_images, val_labels):
                img_np = cv2.imread(str(img))
                h, w = img_np.shape[:2]

                gt_masks = load_yolo_segment_masks(lbl, (h, w))
                gt_masks_all.append(gt_masks)

            return self.metric.evaluate_dataset(
                gt_masks_all,
                pred_masks_all_out
            )

    def extract_all_nouns(self, text: str):
        text = re.sub(r"`{3,}.*", "", text, flags=re.DOTALL).strip()
        doc = nlp(text)
        return [
            token.lemma_.lower()
            for token in doc
            if token.pos_ in {"NOUN", "PROPN"} and token.is_alpha
        ]

    def clean_trailing_backticks(self, text: str) -> str:
        return re.sub(r"\s*`{3,}.*$", "", text).strip()

    # --------------------------------------------------
    # Optimization loop
    # --------------------------------------------------
    def optimize(
        self,
        train_images,
        train_yolo_labels,
        val_images,
        val_yolo_labels,
        workdir,
    ):
        workdir.mkdir(parents=True, exist_ok=True)

        # === visualize train annotations ===
        vis_imgs = []
        for i, (img, lbl) in enumerate(zip(train_images, train_yolo_labels)):
            out_img = workdir / f"train_vis_{i}.png"
            self._visualize_annotations(img, lbl, out_img)
            vis_imgs.append(out_img)

        vis_imgs_str = [str(p) for p in vis_imgs]

        # === optimization ===
        for it in range(self.max_iters):
            if it == 0:
                prompt = PromptGenerator.build_initial(
                    vis_imgs_str,
                    max_words=self.max_desc_words
                )
            else:
                prompt = PromptGenerator.build_iterative(
                    vis_imgs_str,
                    self.history_topk,
                    max_words=self.max_desc_words
                )

            desc = self.llm.inference(prompt, vis_imgs_str, self.max_desc_words)

            # desc and nouns from desc for checking
            for j, desc_for_check in enumerate([self.clean_trailing_backticks(desc)] + self.extract_all_nouns(desc)):

                pred_masks = self._predict_masks(val_images, desc_for_check)
                score = self._evaluate(
                    val_images,
                    val_yolo_labels,
                    pred_masks
                )

                self.history_all.append({
                    "iter": f"{it + 1}.{j}",
                    "desc": desc_for_check,
                    "iou": score,
                })

                self.history_topk = sorted(
                    self.history_all,
                    key=lambda x: x["iou"],
                    reverse=True
                )

                print(
                    f"[Iter {it + 1}.{j}] ({self.task}) {desc} → score={score:.4f}"
                )

        with open(workdir / "history.json", "w", encoding="utf-8") as f:
            json.dump(self.history_all, f, indent=4)

        return max(self.history_all, key=lambda x: x["iou"])


if __name__ == "__main__":

    data_root = Path("C:/projects/labelcv/data/prompt_iter_improver/")

    train_images, train_labels = load_images_and_labels(data_root / "train/images", data_root / "train/labels")
    val_images, val_labels = load_images_and_labels(data_root / "val/images", data_root / "val/labels")

    metric = DetectionAP50()

    optimizer = SAM3PromptOptimizer(
        sam_model_path="resources/sam3.pt",
        llm_model="gemma3:latest",
        metric=metric,
        max_iters=25,
    )

    best = optimizer.optimize(
        train_images=train_images,
        train_yolo_labels=train_labels,
        val_images=val_images,
        val_yolo_labels=val_labels,
        workdir=Path("runs/prompt_opt"),
    )

    print("BEST PROMPT:")
    print(best["desc"])
    print("AP50:", best["iou"])

    best_prompt_file = Path("runs/prompt_opt/best_prompt.txt")
    with open(best_prompt_file, "w", encoding="utf-8") as f:
        f.write(f"BEST PROMPT:\n{best['desc']}\nAP50: {best['iou']:.4f}")
