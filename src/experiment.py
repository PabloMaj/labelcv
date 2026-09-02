import os
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR" # Ignore OOM warning in PyTorch (BUG)

import random
import shutil
import yaml
import numpy as np
import torch
import gc

from datetime import datetime
from pathlib import Path
from enum import Enum
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
from src.utv2 import UnbiasedTeacherV2Trainer


def cuda_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

SUBDIRS_SHARED_STAGE1 = ["weak_yolo", "weak_yolo_kd", "weak_yolo_ub"] # Order is important - first WEAK YOLO (PPL+YOLO)
SUBDIRS_REMOVE_AFTER_EVAL = ["unlabeled", "labeled", "mixed", "pseudo", "val", "_tmp_ignore"]

class ExperimentMode(Enum):
    STANDARD = "standard"
    WEAK_YOLO_PSEUDO = "weak_yolo_pseudo"
    SAM3_PSEUDO = "sam3_pseudo"
    WEAK_YOLO_KD = "weak_yolo_kd"
    WEAK_YOLO_UNBIASED_TEACHER = "weak_yolo_unbiased_teacher"

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
            results = model.val(data=str(data_yaml), workers=0, cache='ram', save_json=True, plots=True, project=work_dir, name="stage1_conf_score_thresh")
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
            elif mode == ExperimentMode.WEAK_YOLO_KD:
                result, work_dir = self._run_knowledge_distillation(repeat_idx=i, **kwargs)
            elif mode == ExperimentMode.WEAK_YOLO_UNBIASED_TEACHER:
                result, work_dir = self._run_unbiased_teacher(repeat_idx=i, **kwargs)
            else:
                raise ValueError(f"Unsupported mode: {mode}")

            if result is None or work_dir is None: continue

            metadata = dict(
                mode=mode.value,
                repeat_idx=i,
                seed=seed,
                model_weights=self.model_weights,
                results=result
            )
            fields = ["target_objects_train", "target_objects_val", "llm_model", "max_iters_prompt_optimizer",
                      "max_desc_words", "train_kwargs_weak", "train_kwargs_normal", "test_kwargs"]
            for f in fields:
                if f in kwargs:
                    metadata[f] = kwargs[f],

            self._save_metadata(work_dir, metadata)
            all_results.append(result)

        return all_results
    
    def _run_standard(self, train_kwargs_weak: dict, train_kwargs_normal: dict, test_kwargs: dict, repeat_idx: int = 0):
        self.logger.info("STANDARD: training on full dataset")

        work_dir = self.output_root / "standard_yolo" / f"repeat_{repeat_idx}_seed_{self.random_state + repeat_idx}"

        work_dir.mkdir(parents=True, exist_ok=True)

        check_path = work_dir / "eval_test"
        if check_path.exists():
            self.logger.info(f"Experiment {check_path} already exists!")
            return None, None

        # ----------------------------------------------------
        # (1) Training model on whole dataset (train + val)
        # ----------------------------------------------------
        self.logger.info("(1) Training model on whole dataset (train + val)")

        trainer = YOLOTrainer(
            model_weights=self.model_weights,
            data_yaml=str(self.data_yaml),
            output_dir=work_dir / "train"
        )

        _ = trainer.train(**train_kwargs_normal)
        final_weights = trainer.final_weights

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

        test_results = evaluator.test(**test_kwargs)       

        del evaluator
        cuda_cleanup()

        return test_results.results_dict, work_dir

    def _run_knowledge_distillation(
            self,
            target_objects_train: int,
            target_objects_val: int,
            train_kwargs_weak: dict,
            train_kwargs_normal: dict, 
            test_kwargs: dict,
            repeat_idx: int = 0
        ):
            self.logger.info("KNOWLEDGE DISTILLATION")
    
            work_dir = self.output_root / "weak_yolo_kd" / f"obj_{target_objects_train}" / f"repeat_{repeat_idx}_seed_{self.random_state + repeat_idx}"
            work_dir.mkdir(parents=True, exist_ok=True)

            check_path = work_dir / "eval_test"
            if check_path.exists():
                self.logger.info(f"Experiment {check_path} already exists!")
                return None, None
    
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
    
            # Check if weak weights exist in different experiment (weak_yolo, weak_yolo_kd or weak_yolo_ub) and reuse them if available
            for subdir in SUBDIRS_SHARED_STAGE1:
                tmp_dir = self.output_root / subdir / f"obj_{target_objects_train}" / f"repeat_{repeat_idx}_seed_{self.random_state + repeat_idx}" / "stage1/weights/best.pt"
                tmp_dir2 = self.output_root / subdir / f"obj_{target_objects_train}" / f"repeat_{repeat_idx}_seed_{self.random_state + repeat_idx}" / "stage1/results.png"

                if tmp_dir.exists() and tmp_dir2.exists():
                    weak_weights = tmp_dir
                    self.logger.info(f"Reusing weak weights from {tmp_dir}")
                    break
            else:
                trainer1 = YOLOTrainer(
                    model_weights=self.model_weights,
                    data_yaml=str(labeled_yaml),
                    output_dir=work_dir / "stage1"
                )
        
                _ = trainer1.train(**train_kwargs_weak)
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
                output_dir=work_dir / "stage2",
            )

            _ = trainer2.train_distill(
                distill_model=str(weak_weights),
                dis=6.0,
                **train_kwargs_normal,
            )

            final_weights = trainer2.final_weights

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
    
            test_results = evaluator.test(**test_kwargs)

            # ----------------------------------------------------
            # (7) Cleanup temporary folders
            # ----------------------------------------------------
            self.logger.info("(7) Cleanup temporary folders")
            for subdir in SUBDIRS_REMOVE_AFTER_EVAL:
                dir_path = work_dir / subdir
                if dir_path.exists():
                    shutil.rmtree(dir_path)

            del evaluator
            cuda_cleanup()
    
            return test_results.results_dict, work_dir
    
    def _run_unbiased_teacher(
            self,
            target_objects_train: int,
            target_objects_val: int,
            train_kwargs_weak: dict,
            train_kwargs_normal: dict, 
            test_kwargs: dict,
            repeat_idx: int = 0
        ):
            self.logger.info("WEAK YOLO PSEUDO-LABELING")
    
            work_dir = self.output_root / "weak_yolo_ub" / f"obj_{target_objects_train}" / f"repeat_{repeat_idx}_seed_{self.random_state + repeat_idx}"
            work_dir.mkdir(parents=True, exist_ok=True)

            check_path = work_dir / "eval_test"
            if check_path.exists():
                self.logger.info(f"Experiment {check_path} already exists!")
                return None, None
    
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
                path: {work_dir.as_posix()}
                train: {train_path.as_posix()}
                val: {val_path.as_posix()}
    
                nc: {self.nc}
                names: {self.class_names}
                """
            )

            # Check if weak weights exist in different experiment (weak_yolo, weak_yolo_kd or weak_yolo_ub) and reuse them if available
            for subdir in SUBDIRS_SHARED_STAGE1:
                tmp_dir = self.output_root / subdir / f"obj_{target_objects_train}" / f"repeat_{repeat_idx}_seed_{self.random_state + repeat_idx}" / "stage1/weights/best.pt"
                tmp_dir2 = self.output_root / subdir / f"obj_{target_objects_train}" / f"repeat_{repeat_idx}_seed_{self.random_state + repeat_idx}" / "stage1/results.png"
            
                if tmp_dir.exists() and tmp_dir2.exists():
                    weak_weights = tmp_dir
                    self.logger.info(f"Reusing weak weights from {tmp_dir}")
                    break
            else:
                trainer1 = YOLOTrainer(
                    model_weights=self.model_weights,
                    data_yaml=str(labeled_yaml),
                    output_dir=work_dir / "stage1"
                )
        
                _ = trainer1.train(**train_kwargs_weak)
                weak_weights = work_dir / "stage1/weights/best.pt"
        
                del trainer1
                cuda_cleanup()

    
            # ----------------------------------------------------
            # (4) Training with Unbiased Teacher (UT)
            # ---------------------------------------------------- 

            self.logger.info("(4) Training with Unbiased Teacher (UT)")

            trainer2 = UnbiasedTeacherV2Trainer(
                model_weights=weak_weights,
                output_dir=work_dir / "stage2",
                config=None,
            )

            # Use folder-based adapter to build simple dataloaders from the
            # mixed image/label folders and unlabeled images.
            _ = trainer2.train_from_dirs(
                labeled_yaml=str(labeled_yaml),             
                unlabeled_images_dir=unlabeled_img_dir_train,
                epochs=train_kwargs_weak.get("epochs", None),
                batch_size=train_kwargs_weak.get("batch", None),
                val_batch_size=test_kwargs.get("batch", None),
                img_size=train_kwargs_weak.get("imgsz", None),
                patience=train_kwargs_weak.get("patience", 50)
            )
            final_weights = trainer2.final_weights_path
    
            del trainer2
            cuda_cleanup()
    
            # ----------------------------------------------------
            # (5) Final evaluation using test set
            # ----------------------------------------------------
            self.logger.info("(5) Final evaluation using test set")
    
            evaluator = YOLOEvaluator(
                weights_path=str(final_weights),
                data_yaml=str(self.data_yaml),
                output_dir=work_dir / "eval_test",
            )
    
            test_results = evaluator.test(**test_kwargs)

            # ----------------------------------------------------
            # (6) Cleanup temporary folders
            # ----------------------------------------------------
            self.logger.info("(6) Cleanup temporary folders")
            for subdir in SUBDIRS_REMOVE_AFTER_EVAL:
                dir_path = work_dir / subdir
                if dir_path.exists():
                    shutil.rmtree(dir_path)
    
            del evaluator
            cuda_cleanup()
    
            return test_results.results_dict, work_dir  

    def _run_weak_yolo_pseudo(
        self,
        target_objects_train: int,
        target_objects_val: int,
        train_kwargs_weak: dict,
        train_kwargs_normal: dict, 
        test_kwargs: dict,
        repeat_idx: int = 0
    ):
        self.logger.info("WEAK YOLO PSEUDO-LABELING")

        work_dir = self.output_root / "weak_yolo" / f"obj_{target_objects_train}" / f"repeat_{repeat_idx}_seed_{self.random_state + repeat_idx}"
        work_dir.mkdir(parents=True, exist_ok=True)

        check_path = work_dir / "eval_test"
        if check_path.exists():
            self.logger.info(f"Experiment {check_path} already exists!")
            return None, None

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

        # Check if weak weights exist in different experiment (weak_yolo, weak_yolo_kd or weak_yolo_ub) and reuse them if available
        for subdir in SUBDIRS_SHARED_STAGE1:
            tmp_dir = self.output_root / subdir / f"obj_{target_objects_train}" / f"repeat_{repeat_idx}_seed_{self.random_state + repeat_idx}" / "stage1/weights/best.pt"
            tmp_dir2 = self.output_root / subdir / f"obj_{target_objects_train}" / f"repeat_{repeat_idx}_seed_{self.random_state + repeat_idx}" / "stage1/results.png"
        
            if tmp_dir.exists() and tmp_dir2.exists():
                weak_weights = tmp_dir
                self.logger.info(f"Reusing weak weights from {tmp_dir}")
                break
        else:
            trainer1 = YOLOTrainer(
                model_weights=self.model_weights,
                data_yaml=str(labeled_yaml),
                output_dir=work_dir / "stage1"
            )

            _ = trainer1.train(**train_kwargs_weak)
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
            model_weights=str(weak_weights),
            data_yaml=str(mixed_yaml),
            output_dir=work_dir / "stage2"
        )

        _ = trainer2.train(**train_kwargs_normal)
        final_weights = trainer2.final_weights

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

        test_results = evaluator.test(**test_kwargs)

        # ----------------------------------------------------
        # (7) Cleanup temporary folders
        # ----------------------------------------------------
        self.logger.info("(7) Cleanup temporary folders")
        for subdir in SUBDIRS_REMOVE_AFTER_EVAL:
            dir_path = work_dir / subdir
            if dir_path.exists():
                shutil.rmtree(dir_path)

        del evaluator
        cuda_cleanup()

        return test_results.results_dict, work_dir

    def _run_sam3_pseudo(
        self,
        target_objects_train: int,
        target_objects_val: int,
        train_kwargs_weak: dict,
        train_kwargs_normal: dict, 
        test_kwargs: dict,
        llm_model: str,
        max_iters_prompt_optimizer: int,
        max_desc_words: int,
        repeat_idx: int = 0
    ):
        self.logger.info("SAM LABELING")

        work_dir = self.output_root / "sam" / f"obj_{target_objects_train}" / f"repeat_{repeat_idx}_seed_{self.random_state + repeat_idx}"
        work_dir.mkdir(parents=True, exist_ok=True)

        check_path = work_dir / "eval_test"
        if check_path.exists():
            self.logger.info(f"Experiment {check_path} already exists!")
            return None, None

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

        _ = trainer.train(**train_kwargs_normal)
        final_weights = trainer.final_weights

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

        test_results = evaluator.test(**test_kwargs)

        del evaluator
        cuda_cleanup()

        return test_results.results_dict, work_dir