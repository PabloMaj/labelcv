import random
import shutil
import yaml
import numpy as np
import torch
import gc

from datetime import datetime
from pathlib import Path
from enum import Enum
from typing import Dict
from ultralytics import YOLO

from src.dataset_layout import DatasetLayout
from src.dataset_preparation import prepare_labeled_unlabeled_with_object_budget
from src.experiment_logger import ExperimentLogger
from src.loaders import load_images_and_labels
from src.metrics import DetectionAP50, SegmentationAP50
from src.prompt_optimizer import SAM3PromptOptimizer
from src.sam_auto_labeler import SAM3AutoLabeler
from src.sam_model import SamSingleton
from src.task_detection import TaskType, detect_task_from_labels
from src.yolo_train_eval import YOLOTrainer, YOLOEvaluator
from src.yolo_weak_auto_labeler import WeakYoloSingleton, YOLOWeakAutoLabeler


def cuda_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


class ExperimentMode(Enum):
    STANDARD = "standard"
    WEAK_YOLO_PSEUDO = "weak_yolo_pseudo"
    SAM3_PSEUDO = "sam3_pseudo"


class VisionTaskExperiment:
    def __init__(
        self,
        dataset_root: Path,
        data_yaml: str,
        output_root: Path,
        model_name: str,
        random_state: int = 42,
    ):
        # ------------------------------------------------------------------
        # Core configuration
        # ------------------------------------------------------------------
        self.dataset = DatasetLayout(dataset_root)
        self.data_yaml = data_yaml
        self.output_root = output_root
        self.model_name = model_name
        self.random_state = random_state
        self.rng = random.Random(random_state)

        # ------------------------------------------------------------------
        # Logging
        # ------------------------------------------------------------------
        self.logger = ExperimentLogger(self.output_root)

        # ------------------------------------------------------------------
        # Dataset metadata (classes, task type)
        # ------------------------------------------------------------------
        self.class_names, self.nc = self._load_class_metadata(self.data_yaml)
        self.task = detect_task_from_labels(self.dataset.labels("train"))

        # ------------------------------------------------------------------
        # Model resolution (task-aware)
        # ------------------------------------------------------------------
        self.model_weights = self._resolve_model_weights()

        # ------------------------------------------------------------------
        # Evaluation metric (task-aware)
        # ------------------------------------------------------------------
        self.metric = self._resolve_metric()

        # ------------------------------------------------------------------
        # Sanity logs
        # ------------------------------------------------------------------
        self.logger.info(f"Detected task: {self.task.value}")
        self.logger.info(f"Classes ({self.nc}): {self.class_names}")

    def _load_class_metadata(self, data_yaml: Path) -> tuple[dict[int, str], int]:
        with open(data_yaml, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if "names" not in data:
            raise ValueError("data.yaml must contain 'names' field")

        class_names = {int(k): v for k, v in data["names"].items()}
        nc = len(class_names)

        return class_names, nc

    def _resolve_metric(self):
        if self.task == TaskType.DETECT:
            return DetectionAP50()
        if self.task == TaskType.SEGMENT:
            return SegmentationAP50()
        raise ValueError(f"Unsupported task: {self.task}")

    def _resolve_model_weights(self) -> str:
        if self.task == TaskType.DETECT:
            return f"{self.model_name}.pt"
        if self.task == TaskType.SEGMENT:
            return f"{self.model_name}-seg.pt"
        raise ValueError(f"Unsupported task: {self.task}")

    def _save_metadata(self, work_dir: Path, metadata: dict):
        metadata = dict(metadata)
        metadata["timestamp"] = datetime.now().isoformat()
        metadata["random_state"] = self.random_state

        meta_path = work_dir / "metadata.yaml"
        with open(meta_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(metadata, f, sort_keys=False)

    def _find_best_conf_threshold(self, weights: Path, data_yaml: Path, work_dir: Path) -> float:

        model = YOLO(weights)
        with torch.no_grad():
            results = model.val(data=str(data_yaml), save_json=True, plots=True, project=work_dir, name="stage1_conf_score_thresh")
        f1_curve = results.box.f1_curve.flatten()
        best_idx = f1_curve.argmax()
        conf_thresholds = np.linspace(0, 1, len(f1_curve))
        best_conf = conf_thresholds[best_idx]
        self.logger.info(f"best_conf={best_conf}")

        del results
        del model
        cuda_cleanup()

        return best_conf

    def run(self, mode: ExperimentMode, repeats: int = 1, **kwargs):
        self.logger.info(f"Running mode: {mode.value}, repeats={repeats}")

        all_results = []

        for i in range(repeats):
            seed = self.random_state + i
            self.rng.seed(seed)

            self.logger.info(f"=== REPEAT {i + 1}/{repeats} (seed={seed}) ===")

            cuda_cleanup()

            if mode == ExperimentMode.STANDARD:
                result, work_dir = self._run_standard(repeat_idx=i, **kwargs)
            elif mode == ExperimentMode.WEAK_YOLO_PSEUDO:
                result, work_dir = self._run_weak_yolo_pseudo(repeat_idx=i, **kwargs)
            elif mode == ExperimentMode.SAM3_PSEUDO:
                result, work_dir = self._run_sam3_pseudo(repeat_idx=i, **kwargs)
            else:
                raise ValueError(f"Unsupported mode: {mode}")

            metadata = dict(
                mode=mode.value,
                repeat_idx=i,
                seed=seed,
                model_weights=self.model_weights,
                results=result
            )
            fields = ["target_objects_train", "target_objects_val", "llm_model", "max_iters_prompt_optimizer",
                      "max_desc_words", "train_kwargs", "test_kwargs"]
            for f in fields:
                if f in kwargs:
                    metadata[f] = kwargs[f],

            self._save_metadata(work_dir, metadata)
            all_results.append(result)

        return all_results

    def _run_standard(self, train_kwargs: dict, test_kwargs: dict, repeat_idx: int = 0):
        self.logger.info("STANDARD: training on full dataset")

        work_dir = self.output_root / "standard_yolo" / f"repeat_{repeat_idx}_seed_{self.random_state + repeat_idx}"

        work_dir.mkdir(parents=True, exist_ok=True)

        # ----------------------------------------------------
        # (1) Training model on whole dataset (train + val)
        # ----------------------------------------------------
        self.logger.info("(1) Training model on whole dataset (train + val)")

        trainer = YOLOTrainer(
            model_weights=self.model_weights,
            data_yaml=str(self.data_yaml),
            output_dir=work_dir / "train"
        )

        _ = trainer.train(**train_kwargs, project=work_dir, name="train")
        final_weights = work_dir / "train/weights/best.pt"

        del trainer
        cuda_cleanup()

        # ----------------------------------------------------
        # (2) Final evaluation using test set
        # ----------------------------------------------------
        self.logger.info("(2) Final evaluation using test set")

        evaluator = YOLOEvaluator(
            weights_path=str(final_weights),
            data_yaml=str(self.data_yaml),
            output_dir=work_dir / "eval_test",
        )

        test_results = evaluator.test(**test_kwargs, project=work_dir, name="eval_test")

        del evaluator
        cuda_cleanup()

        return test_results.results_dict, work_dir

    def _run_weak_yolo_pseudo(
        self,
        target_objects_train: int,
        target_objects_val: int,
        train_kwargs: Dict,
        test_kwargs: Dict,
        repeat_idx: int = 0
    ):
        self.logger.info("WEAK YOLO PSEUDO-LABELING")

        work_dir = self.output_root / "weak_yolo" / f"obj_{target_objects_train}" / f"repeat_{repeat_idx}_seed_{self.random_state + repeat_idx}"
        work_dir.mkdir(parents=True, exist_ok=True)

        # ----------------------------------------------------
        # (1) Prepare sets of labeled and unlabeled samples for training
        # ----------------------------------------------------
        self.logger.info("(1) Prepare sets of labeled and unlabeled samples for training")

        labeled_img_dir_train = work_dir / "labeled/images"
        labeled_lbl_dir_train = work_dir / "labeled/labels"
        unlabeled_img_dir_train = work_dir / "unlabeled/images"

        train_imgs = sorted(self.dataset.images("train").glob("*.*"))

        stats_train = prepare_labeled_unlabeled_with_object_budget(
            images=train_imgs,
            labels_dir=self.dataset.labels("train"),
            labeled_img_dir=labeled_img_dir_train,
            labeled_lbl_dir=labeled_lbl_dir_train,
            unlabeled_img_dir=unlabeled_img_dir_train,
            target_objects=target_objects_train,
            rng=self.rng,
            mask_mode="mean",
            task=self.task.value
        )
        print(stats_train)

        # ----------------------------------------------------
        # (2) Prepare set of labeled samples for validation
        # ----------------------------------------------------
        self.logger.info("(2) Prepare set of labeled samples for validation")

        labeled_img_dir_val = work_dir / "val/images"
        labeled_lbl_dir_val = work_dir / "val/labels"
        unlabeled_img_val = work_dir / "_tmp_ignore"

        val_imgs = sorted(self.dataset.images("val").glob("*.*"))

        stats_val = prepare_labeled_unlabeled_with_object_budget(
            images=val_imgs,
            labels_dir=self.dataset.labels("val"),
            labeled_img_dir=labeled_img_dir_val,
            labeled_lbl_dir=labeled_lbl_dir_val,
            unlabeled_img_dir=unlabeled_img_val,
            target_objects=target_objects_val,
            rng=self.rng,
            mask_mode="mean",
            task=self.task.value
        )
        print(stats_val)

        # ----------------------------------------------------
        # (3) First train with small set of labeled samples
        # ----------------------------------------------------
        self.logger.info("(3) First train with small set of labeled samples")

        train_path = labeled_img_dir_train.relative_to(work_dir)
        val_path = labeled_img_dir_val.relative_to(work_dir)

        labeled_yaml = work_dir / "labeled.yaml"
        labeled_yaml.write_text(
            f"""
            train: {train_path.as_posix()}
            val: {val_path.as_posix()}

            nc: {self.nc}
            names: {self.class_names}
            """
        )

        trainer1 = YOLOTrainer(
            model_weights=self.model_weights,
            data_yaml=str(labeled_yaml),
            output_dir=work_dir / "stage1"
        )

        _ = trainer1.train(**train_kwargs, project=work_dir, name="stage1")
        weak_weights = work_dir / "stage1/weights/best.pt"

        del trainer1
        cuda_cleanup()

        # additional validation to extract optimal working point (maximazing F1-score)
        best_conf = self._find_best_conf_threshold(weights=weak_weights, data_yaml=labeled_yaml, work_dir=work_dir)

        # ----------------------------------------------------
        # (4) Pseudo labeling pool of unlabeled samples
        # ----------------------------------------------------
        self.logger.info("(4) Pseudo labeling pool of unlabeled samples")

        weak_yolo = WeakYoloSingleton(
            model_path=str(weak_weights),
            conf_thresh=best_conf,
        )

        labeler = YOLOWeakAutoLabeler(
            yolo=weak_yolo,
            class_id=0,
            conf_thresh=best_conf,
            task=self.task.value
        )

        pseudo_lbl_dir = work_dir / "pseudo/labels"
        pseudo_lbl_dir.mkdir(parents=True, exist_ok=True)

        labeler.label_folder(
            image_dir=unlabeled_img_dir_train,
            out_label_dir=pseudo_lbl_dir,
        )

        del labeler
        del weak_yolo
        WeakYoloSingleton._instance = None
        cuda_cleanup()

        # ----------------------------------------------------
        # (5) Mixed train with samples with labels and pseudo-labels
        # ----------------------------------------------------
        self.logger.info("(5) Mixed train with samples with labels and pseudo-labels")

        mixed_img_dir = work_dir / "mixed/images"
        mixed_lbl_dir = work_dir / "mixed/labels"
        mixed_img_dir.mkdir(parents=True, exist_ok=True)
        mixed_lbl_dir.mkdir(parents=True, exist_ok=True)

        for img in labeled_img_dir_train.glob("*"):
            shutil.copy(img, mixed_img_dir / img.name)
            shutil.copy(
                labeled_lbl_dir_train / f"{img.stem}.txt",
                mixed_lbl_dir / f"{img.stem}.txt",
            )

        for img in unlabeled_img_dir_train.glob("*"):
            lbl = pseudo_lbl_dir / f"{img.stem}.txt"
            if lbl.exists():
                shutil.copy(img, mixed_img_dir / img.name)
                shutil.copy(lbl, mixed_lbl_dir / lbl.name)

        mixed_path = mixed_img_dir.relative_to(work_dir)
        val_path = labeled_img_dir_val.relative_to(work_dir)

        mixed_yaml = work_dir / "mixed.yaml"
        mixed_yaml.write_text(
            f"""
            train: {mixed_path.as_posix()}
            val: {val_path.as_posix()}

            nc: {self.nc}
            names: {self.class_names}
            """
        )

        trainer2 = YOLOTrainer(
            model_weights=self.model_weights,
            data_yaml=str(mixed_yaml),
            output_dir=work_dir / "stage2"
        )

        _ = trainer2.train(**train_kwargs, project=work_dir, name="stage2")
        final_weights = work_dir / "stage2/weights/best.pt"

        del trainer2
        cuda_cleanup()

        # ----------------------------------------------------
        # (6) Final evaluation using test set
        # ----------------------------------------------------
        self.logger.info("(6) Final evaluation using test set")

        evaluator = YOLOEvaluator(
            weights_path=str(final_weights),
            data_yaml=str(self.data_yaml),
            output_dir=work_dir / "eval_test",
        )

        test_results = evaluator.test(**test_kwargs, project=work_dir, name="eval_test")

        del evaluator
        cuda_cleanup()

        return test_results.results_dict, work_dir

    def _run_sam3_pseudo(
        self,
        target_objects_train: int,
        target_objects_val: int,
        train_kwargs: Dict,
        test_kwargs: Dict,
        llm_model: str,
        max_iters_prompt_optimizer: int,
        max_desc_words: int,
        repeat_idx: int = 0
    ):
        self.logger.info("SAM LABELING")

        work_dir = self.output_root / "sam" / f"obj_{target_objects_train}" / f"repeat_{repeat_idx}_seed_{self.random_state + repeat_idx}"
        work_dir.mkdir(parents=True, exist_ok=True)

        # ----------------------------------------------------
        # (1) Prepare sets of labeled and unlabeled samples for training
        # ----------------------------------------------------
        self.logger.info("(1) Prepare sets of labeled and unlabeled samples for training")

        train_imgs = sorted(self.dataset.images("train").glob("*.*"))

        labeled_img_dir_train = work_dir / "labeled/images"
        labeled_lbl_dir_train = work_dir / "labeled/labels"
        unlabeled_img_dir_train = work_dir / "unlabeled/images"

        stats_train = prepare_labeled_unlabeled_with_object_budget(
            images=train_imgs,
            labels_dir=self.dataset.labels("train"),
            labeled_img_dir=labeled_img_dir_train,
            labeled_lbl_dir=labeled_lbl_dir_train,
            unlabeled_img_dir=unlabeled_img_dir_train,
            target_objects=target_objects_train,
            rng=self.rng,
            mask_mode="mean",
            task=self.task.value
        )
        print(stats_train)

        # ----------------------------------------------------
        # (2) Prepare set of labeled samples for validation
        # ----------------------------------------------------
        self.logger.info("(2) Prepare set of labeled samples for validation")

        val_imgs = sorted(self.dataset.images("val").glob("*.*"))

        labeled_img_dir_val = work_dir / "val/images"
        labeled_lbl_dir_val = work_dir / "val/labels"
        unlabeled_img_val = work_dir / "_tmp_ignore"

        stats_val = prepare_labeled_unlabeled_with_object_budget(
            images=val_imgs,
            labels_dir=self.dataset.labels("val"),
            labeled_img_dir=labeled_img_dir_val,
            labeled_lbl_dir=labeled_lbl_dir_val,
            unlabeled_img_dir=unlabeled_img_val,
            target_objects=target_objects_val,
            rng=self.rng,
            mask_mode="mean",
            task=self.task.value
        )
        print(stats_val)

        # ----------------------------------------------------
        # (3) Determine best description for marked objects
        # ----------------------------------------------------
        train_images, train_labels = load_images_and_labels(labeled_img_dir_train, labeled_lbl_dir_train)
        val_images, val_labels = load_images_and_labels(labeled_img_dir_val, labeled_lbl_dir_val)

        optimizer = SAM3PromptOptimizer(
            sam_model_path="resources/sam3.pt",
            llm_model=llm_model,
            task=self.task.value,
            max_iters=max_iters_prompt_optimizer,
            max_desc_words=max_desc_words,
        )

        prompt_optimizer_dir = work_dir / "prompt_optimizer"
        prompt_optimizer_dir.mkdir(parents=True, exist_ok=True)

        best = optimizer.optimize(
            train_images=train_images,
            train_yolo_labels=train_labels,
            val_images=val_images,
            val_yolo_labels=val_labels,
            workdir=prompt_optimizer_dir
        )

        del optimizer
        cuda_cleanup()

        self.logger.info(f"BEST PROMPT:\n{best['desc']}\nAP50: {best['iou']:.4f}")
        best_prompt_file = work_dir / "best_prompt.txt"
        with open(best_prompt_file, "w", encoding="utf-8") as f:
            f.write(f"BEST PROMPT:\n{best['desc']}\nAP50: {best['iou']:.4f}")

        prompt_filter = best['desc']
        print(f"prompt_filter={prompt_filter}")

        # ----------------------------------------------------
        # (4) Pseudo labeling pool of unlabeled samples with SAM
        # ----------------------------------------------------
        self.logger.info("(4) Pseudo labeling pool of unlabeled samples with SAM")

        model_sam = SamSingleton(model_path="resources/sam3.pt",)
        labeler = SAM3AutoLabeler(sam=model_sam, task=self.task.value, class_id=0)

        pseudo_lbl_dir = work_dir / "pseudo/labels"
        pseudo_lbl_dir.mkdir(parents=True, exist_ok=True)

        labeler.label_folder(
            image_dir=unlabeled_img_dir_train,
            out_label_dir=pseudo_lbl_dir,
            prompt_filter=prompt_filter,
            visualize=False
        )

        del labeler
        del model_sam
        SamSingleton._instance = None
        cuda_cleanup()

        # ----------------------------------------------------
        # (5) Mixed train with samples with labels and pseudo-labels
        # ----------------------------------------------------
        self.logger.info("(5) Mixed train with samples with labels and pseudo-labels")

        mixed_img_dir = work_dir / "mixed/images"
        mixed_lbl_dir = work_dir / "mixed/labels"
        mixed_img_dir.mkdir(parents=True, exist_ok=True)
        mixed_lbl_dir.mkdir(parents=True, exist_ok=True)

        for img in labeled_img_dir_train.glob("*"):
            shutil.copy(img, mixed_img_dir / img.name)
            shutil.copy(
                labeled_lbl_dir_train / f"{img.stem}.txt",
                mixed_lbl_dir / f"{img.stem}.txt",
            )

        for img in unlabeled_img_dir_train.glob("*"):
            lbl = pseudo_lbl_dir / f"{img.stem}.txt"
            if lbl.exists():
                shutil.copy(img, mixed_img_dir / img.name)
                shutil.copy(lbl, mixed_lbl_dir / lbl.name)

        mixed_path = mixed_img_dir.relative_to(work_dir)
        val_path = labeled_img_dir_val.relative_to(work_dir)

        mixed_yaml = work_dir / "mixed.yaml"
        mixed_yaml.write_text(
            f"""
            train: {mixed_path.as_posix()}
            val: {val_path.as_posix()}

            nc: {self.nc}
            names: {self.class_names}
            """
        )

        trainer = YOLOTrainer(
            model_weights=self.model_weights,
            data_yaml=str(mixed_yaml),
            output_dir=work_dir / "yolo_train"
        )

        _ = trainer.train(**train_kwargs, project=work_dir, name="yolo_train")
        final_weights = work_dir / "yolo_train/weights/best.pt"

        del trainer
        cuda_cleanup()

        # ----------------------------------------------------
        # (6) Final evaluation using test set
        # ----------------------------------------------------
        self.logger.info("(6) Final evaluation using test set")

        evaluator = YOLOEvaluator(
            weights_path=str(final_weights),
            data_yaml=str(self.data_yaml),
            output_dir=work_dir / "eval_test",
        )

        test_results = evaluator.test(**test_kwargs, project=work_dir, name="eval_test")

        del evaluator
        cuda_cleanup()

        return test_results.results_dict, work_dir


if __name__ == "__main__":

    REPEATS = 3
    DATASET_NAMES = ["bee_subset_1", "african-wildlife_buffalo", "african-wildlife_elephant", "african-wildlife_rhino",
                     "african-wildlife_zebra", "construction-ppe_gloves", "construction-ppe_goggles", "construction-ppe_helmet",
                     "construction-ppe_vest", "crack-seg", "wheat_arvalis_1", "wheat_arvalis_3", "wheat_ethz_1", "wheat_rres_1",
                     "homeobjects_chair", "homeobjects_table", "homeobjects_potted_plant", "homeobjects_lamp"]
    print(f"No of datasets: {len(DATASET_NAMES)}")

    train_kwargs = dict(
        epochs=200,
        imgsz=640,
        batch=16,
        patience=20
    )

    test_kwargs = dict(
        imgsz=640,
    )

    for dataset_name in DATASET_NAMES:

        output_root = Path(f"runs/{dataset_name}")
        data_yaml = f"data/datasets/{dataset_name}/data.yaml"

        exp = VisionTaskExperiment(
            dataset_root=Path(f"data/datasets/{dataset_name}"),
            data_yaml=data_yaml,
            model_name="yolo11n",
            output_root=output_root,
            random_state=42
        )

        """
        # STANDARD
        results = exp.run(
            mode=ExperimentMode.STANDARD,
            train_kwargs=train_kwargs,
            test_kwargs=test_kwargs,
            repeats=REPEATS
        )
        """

        # PSEUDO WEAK YOLO
        weak_target_objects_train_values = [40]
        for target_objects_train in weak_target_objects_train_values:
            print(f"\n=== WEAK YOLO | target_objects_train={target_objects_train} | repeats={REPEATS} ===")
            results = exp.run(
                mode=ExperimentMode.WEAK_YOLO_PSEUDO,
                target_objects_train=target_objects_train,
                target_objects_val=target_objects_train,
                train_kwargs=train_kwargs,
                test_kwargs=test_kwargs,
                repeats=REPEATS
            )

        """
        # PSEUDO SAM
        sam_target_objects_train_values = [20]
        for target_objects_train in sam_target_objects_train_values:
            print(f"\n=== SAM | target_objects_train={target_objects_train} | repeats={REPEATS} ===")
            results = exp.run(
                mode=ExperimentMode.SAM3_PSEUDO,
                target_objects_train=target_objects_train,
                target_objects_val=target_objects_train,
                train_kwargs=train_kwargs,
                test_kwargs=test_kwargs,
                repeats=REPEATS,
                llm_model="ministral-3:latest",
                max_iters_prompt_optimizer=50,
                max_desc_words=5
            )
        """
