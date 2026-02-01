
import cv2
import json
from pathlib import Path

from src.prompt_generator import PromptGenerator
from src.llm_model import OllamaVisionLLM
from src.sam_model import SamSingleton
from src.yolo_bbox_visualizer import YoloBBoxVisualizer
from src.utils import mask_to_bbox, load_yolo_bboxes, masks_to_image_space, load_images_and_labels, DetectionAP50


class SAM3PromptOptimizer:
    def __init__(
        self,
        sam_model_path: str,
        llm_model: str,
        metric: DetectionAP50,
        max_iters: int = 5,
        max_desc_words: int = 10
    ):
        self.sam = SamSingleton(model_path=sam_model_path)
        self.llm = OllamaVisionLLM(llm_model)
        self.metric = metric
        self.max_iters = max_iters
        self.max_desc_words = max_desc_words
        self.history_all = []
        self.history_topk = []
        self.top_k = 3

    def _predict_masks(self, images, prompt):
        preds = []
        for img in images:
            results = self.sam.predict(
                path_to_image=str(img),
                prompt_filter=prompt
            )
            preds.append(results[0].masks)
        return preds

    def optimize(
        self,
        train_images,
        train_yolo_labels,
        val_images,
        val_yolo_labels,
        workdir,
    ):
        workdir.mkdir(parents=True, exist_ok=True)

        # ==========================
        # TRAIN → visualization
        # ==========================
        boxed_imgs = []
        for i, (img, lbl) in enumerate(zip(train_images, train_yolo_labels)):
            out_img = workdir / f"train_boxed_{i}.png"
            YoloBBoxVisualizer.draw_red_boxes(img, lbl, out_img)
            boxed_imgs.append(out_img)

        boxed_str = [str(p) for p in boxed_imgs]

        # ==========================
        # ITERATIONS
        # ==========================
        for it in range(self.max_iters):
            if it == 0:
                prompt = PromptGenerator.build_initial(boxed_str, max_words=self.max_desc_words)
            else:
                prompt = PromptGenerator.build_iterative(boxed_str, self.history_topk, max_words=self.max_desc_words)
            desc = self.llm.inference(prompt, boxed_str)
            previous_descs = {h["desc"] for h in self.history_topk}
            if desc in previous_descs:
                retry_prompt = (
                    prompt + "\n\nThe generated description was too similar to the best previous ones. "
                    "Generate a different description."
                )
                desc = self.llm.inference(retry_prompt, boxed_str)

            # ==========================
            # VAL → METRICS
            # ==========================
            pred_masks = self._predict_masks(val_images, desc)

            pred_boxes_all = []
            gt_boxes_all = []

            for img, lbl, masks in zip(val_images, val_yolo_labels, pred_masks):
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

            ap50 = self.metric.evaluate_dataset(
                gt_boxes_all, pred_boxes_all
            )

            self.history_all.append({
                "iter": it + 1,
                "desc": desc,
                "iou": ap50
            })

            self.history_topk = sorted(
                self.history_all,
                key=lambda x: x["iou"],
                reverse=True
            )[:self.top_k]

            print(f"[Iter {it + 1}] {desc} → AP50={ap50:.4f}")

        history_file = workdir / "history.json"
        with open(history_file, "w", encoding="utf-8") as f:
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
