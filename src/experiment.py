import random
import shutil
import yaml
import numpy as np
from datetime import datetime
import traceback

from pathlib import Path
from enum import Enum
from typing import Dict

from ultralytics import YOLO

from src.utils import DetectionAP50, load_images_and_labels
from src.dataset_layout import DatasetLayout
from src.experiment_logger import ExperimentLogger
from src.yolo_train_eval import YOLOTrainer, YOLOEvaluator
from src.yolo_weak_auto_labeler import WeakYoloSingleton, YOLOWeakAutoLabeler
from src.prompt_optimizer import SAM3PromptOptimizer
from src.sam_model import SamSingleton
from src.sam_auto_labeler import SAM3AutoLabeler


class ExperimentMode(Enum):
    STANDARD = "standard"
    WEAK_YOLO_PSEUDO = "weak_yolo_pseudo"
    SAM3_PSEUDO = "sam3_pseudo"


class ObjectDetectionExperiment:
    def __init__(
        self,
        dataset_root: Path,
        data_yaml: str,
        output_root: Path,
        model_type: str,
        random_state: int = 42,
    ):
        self.dataset = DatasetLayout(dataset_root)
        self.data_yaml = data_yaml
        self.output_root = output_root
        self.model_type = model_type
        self.random_state = random_state
        self.rng = random.Random(random_state)

        # ----------------------------------
        # READ nc + class names FROM YAML
        # ----------------------------------
        with open(self.data_yaml, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if "names" not in data:
            raise ValueError("data.yaml must contain 'names' field")

        self.class_names: dict[int, str] = {
            int(k): v for k, v in data["names"].items()
        }
        self.nc: int = len(self.class_names)
        print(f"self.class_names={self.class_names}")
        print(f"self.nc={self.nc}")

        self.logger = ExperimentLogger(output_root)
        self.metric = DetectionAP50()

    def _save_metadata(self, work_dir: Path, metadata: dict):
        metadata = dict(metadata)
        metadata["timestamp"] = datetime.now().isoformat()
        metadata["random_state"] = self.random_state

        meta_path = work_dir / "metadata.yaml"
        with open(meta_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(metadata, f, sort_keys=False)

    def safe_sample(self, population, k, work_dir: Path, description: str):
        try:
            return set(self.rng.sample(population, k))
        except ValueError as e:
            work_dir.mkdir(parents=True, exist_ok=True)
            error_file = work_dir / "error.log"
            with open(error_file, "a", encoding="utf-8") as f:
                f.write(f"\nERROR during sampling {description}:\n")
                f.write(str(e) + "\n\n")
                f.write("Full traceback:\n")
                f.write(traceback.format_exc())
            self.logger.error(f"Sampling error ({description}). See {error_file}")
            return None

    def run(self, mode: ExperimentMode, repeats: int = 1, **kwargs):
        self.logger.info(f"Running mode: {mode.value}, repeats={repeats}")

        all_results = []

        for i in range(repeats):
            seed = self.random_state + i
            self.rng.seed(seed)

            self.logger.info(f"=== REPEAT {i + 1}/{repeats} (seed={seed}) ===")

            if mode == ExperimentMode.STANDARD:
                result, work_dir = self._run_standard(repeat_idx=i, **kwargs)

            elif mode == ExperimentMode.WEAK_YOLO_PSEUDO:
                result, work_dir = self._run_weak_yolo_pseudo(
                    repeat_idx=i, **kwargs
                )

            elif mode == ExperimentMode.SAM3_PSEUDO:
                result, work_dir = self._run_sam3_pseudo(
                    repeat_idx=i, **kwargs
                )

            else:
                raise ValueError(f"Unsupported mode: {mode}")

            metadata = dict(
                mode=mode.value,
                repeat_idx=i,
                seed=seed,
                model_type=self.model_type,
                n_labeled=kwargs.get("n_labeled"),
                n_val=kwargs.get("n_val"),
                train_kwargs=kwargs.get("train_kwargs"),
                test_kwargs=kwargs.get("test_kwargs"),
                results=result
            )

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
            model_type=self.model_type,
            data_yaml=str(data_yaml),
            output_dir=work_dir / "train"
        )

        _ = trainer.train(**train_kwargs, project=work_dir, name="train")
        final_weights = work_dir / "train/weights/best.pt"

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

        return test_results.results_dict, work_dir

    def _run_weak_yolo_pseudo(
        self,
        n_labeled: int,
        n_val: int,
        train_kwargs: Dict,
        test_kwargs: Dict,
        repeat_idx: int = 0
    ):
        self.logger.info("WEAK YOLO PSEUDO-LABELING")

        work_dir = self.output_root / "weak_yolo" / f"n_labeled_{n_labeled}" / f"repeat_{repeat_idx}_seed_{self.random_state + repeat_idx}"
        work_dir.mkdir(parents=True, exist_ok=True)

        # ----------------------------------------------------
        # (1) Prepare sets of labeled and unlabeled samples for training
        # ----------------------------------------------------
        self.logger.info("(1) Prepare sets of labeled and unlabeled samples for training")

        train_imgs = sorted(self.dataset.images("train").glob("*.*"))
        assert n_labeled < len(train_imgs)

        labeled_imgs = self.safe_sample(train_imgs, n_labeled, work_dir, "train images")
        if labeled_imgs is None:
            return {}, work_dir
        unlabeled_imgs = set(train_imgs) - labeled_imgs

        labeled_img_dir = work_dir / "labeled/images"
        labeled_lbl_dir = work_dir / "labeled/labels"
        unlabeled_img_dir = work_dir / "unlabeled/images"

        labeled_img_dir.mkdir(parents=True, exist_ok=True)
        labeled_lbl_dir.mkdir(parents=True, exist_ok=True)
        unlabeled_img_dir.mkdir(parents=True, exist_ok=True)

        for img in labeled_imgs:
            shutil.copy(img, labeled_img_dir / img.name)
            shutil.copy(
                self.dataset.labels("train") / f"{img.stem}.txt",
                labeled_lbl_dir / f"{img.stem}.txt",
            )

        for img in unlabeled_imgs:
            shutil.copy(img, unlabeled_img_dir / img.name)

        # ----------------------------------------------------
        # (2) Prepare set of labeled samples for validation
        # ----------------------------------------------------
        self.logger.info("(2) Prepare set of labeled samples for validation")

        val_imgs = sorted(self.dataset.images("val").glob("*.*"))
        val_sample = self.safe_sample(val_imgs, n_val, work_dir, "validation images")
        if val_sample is None:
            return {}, work_dir

        val_img_dir = work_dir / "val/images"
        val_lbl_dir = work_dir / "val/labels"
        val_img_dir.mkdir(parents=True, exist_ok=True)
        val_lbl_dir.mkdir(parents=True, exist_ok=True)

        for img in val_sample:
            shutil.copy(img, val_img_dir / img.name)
            shutil.copy(
                self.dataset.labels("val") / f"{img.stem}.txt",
                val_lbl_dir / f"{img.stem}.txt",
            )

        # ----------------------------------------------------
        # (3) First train with small set of labeled samples
        # ----------------------------------------------------
        self.logger.info("(3) First train with small set of labeled samples")

        train_path = labeled_img_dir.relative_to(work_dir)
        val_path = val_img_dir.relative_to(work_dir)

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
            model_type=self.model_type,
            data_yaml=str(labeled_yaml),
            output_dir=work_dir / "stage1"
        )

        _ = trainer1.train(**train_kwargs, project=work_dir, name="stage1")
        weak_weights = work_dir / "stage1/weights/best.pt"

        # additional validation to extract optimal working point (maximazing F1-score)
        model = YOLO(weak_weights)
        results = model.val(data=str(labeled_yaml), save_json=True, plots=True, project=work_dir, name="stage1_conf_score_thresh")
        f1_curve = results.box.f1_curve.flatten()
        best_idx = f1_curve.argmax()
        conf_thresholds = np.linspace(0, 1, len(f1_curve))
        best_conf = conf_thresholds[best_idx]
        self.logger.info(f"best_conf={best_conf}")

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
        )

        pseudo_lbl_dir = work_dir / "pseudo/labels"
        pseudo_lbl_dir.mkdir(parents=True, exist_ok=True)

        labeler.label_folder(
            image_dir=unlabeled_img_dir,
            out_label_dir=pseudo_lbl_dir,
        )

        # ----------------------------------------------------
        # (5) Mixed train with samples with labels and pseudo-labels
        # ----------------------------------------------------
        self.logger.info("(5) Mixed train with samples with labels and pseudo-labels")

        mixed_img_dir = work_dir / "mixed/images"
        mixed_lbl_dir = work_dir / "mixed/labels"
        mixed_img_dir.mkdir(parents=True, exist_ok=True)
        mixed_lbl_dir.mkdir(parents=True, exist_ok=True)

        for img in labeled_img_dir.glob("*"):
            shutil.copy(img, mixed_img_dir / img.name)
            shutil.copy(
                labeled_lbl_dir / f"{img.stem}.txt",
                mixed_lbl_dir / f"{img.stem}.txt",
            )

        for img in unlabeled_img_dir.glob("*"):
            lbl = pseudo_lbl_dir / f"{img.stem}.txt"
            if lbl.exists():
                shutil.copy(img, mixed_img_dir / img.name)
                shutil.copy(lbl, mixed_lbl_dir / lbl.name)

        mixed_path = mixed_img_dir.relative_to(work_dir)
        val_path = val_img_dir.relative_to(work_dir)

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
            model_type=self.model_type,
            data_yaml=str(mixed_yaml),
            output_dir=work_dir / "stage2"
        )

        _ = trainer2.train(**train_kwargs, project=work_dir, name="stage2")
        final_weights = work_dir / "stage2/weights/best.pt"

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
        return test_results.results_dict, work_dir

    def _run_sam3_pseudo(
        self,
        n_labeled: int,
        n_val: int,
        train_kwargs: Dict,
        test_kwargs: Dict,
        repeat_idx: int = 0
    ):
        self.logger.info("SAM LABELING")

        work_dir = self.output_root / "sam" / f"n_labeled_{n_labeled}" / f"repeat_{repeat_idx}_seed_{self.random_state + repeat_idx}"
        work_dir.mkdir(parents=True, exist_ok=True)

        # ----------------------------------------------------
        # (1) Prepare sets of labeled and unlabeled samples for training
        # ----------------------------------------------------
        self.logger.info("(1) Prepare sets of labeled and unlabeled samples for training")

        train_imgs = sorted(self.dataset.images("train").glob("*.*"))
        assert n_labeled < len(train_imgs)

        labeled_imgs = self.safe_sample(train_imgs, n_labeled, work_dir, "train images")
        if labeled_imgs is None:
            return {}, work_dir
        unlabeled_imgs = set(train_imgs) - labeled_imgs

        labeled_img_dir = work_dir / "labeled/images"
        labeled_lbl_dir = work_dir / "labeled/labels"
        unlabeled_img_dir = work_dir / "unlabeled/images"

        labeled_img_dir.mkdir(parents=True, exist_ok=True)
        labeled_lbl_dir.mkdir(parents=True, exist_ok=True)
        unlabeled_img_dir.mkdir(parents=True, exist_ok=True)

        for img in labeled_imgs:
            shutil.copy(img, labeled_img_dir / img.name)
            shutil.copy(
                self.dataset.labels("train") / f"{img.stem}.txt",
                labeled_lbl_dir / f"{img.stem}.txt",
            )

        for img in unlabeled_imgs:
            shutil.copy(img, unlabeled_img_dir / img.name)

        # ----------------------------------------------------
        # (2) Prepare set of labeled samples for validation
        # ----------------------------------------------------
        self.logger.info("(2) Prepare set of labeled samples for validation")

        val_imgs = sorted(self.dataset.images("val").glob("*.*"))
        val_sample = self.safe_sample(val_imgs, n_val, work_dir, "validation images")
        if val_sample is None:
            return {}, work_dir

        val_img_dir = work_dir / "val/images"
        val_lbl_dir = work_dir / "val/labels"
        val_img_dir.mkdir(parents=True, exist_ok=True)
        val_lbl_dir.mkdir(parents=True, exist_ok=True)

        for img in val_sample:
            shutil.copy(img, val_img_dir / img.name)
            shutil.copy(
                self.dataset.labels("val") / f"{img.stem}.txt",
                val_lbl_dir / f"{img.stem}.txt",
            )

        # ----------------------------------------------------
        # (3) Determine best description for marked objects
        # ----------------------------------------------------
        train_images, train_labels = load_images_and_labels(labeled_img_dir, labeled_lbl_dir)
        val_images, val_labels = load_images_and_labels(val_img_dir, val_lbl_dir)

        metric = DetectionAP50()

        optimizer = SAM3PromptOptimizer(
            sam_model_path="resources/sam3.pt",
            llm_model="gemma3:latest",
            metric=metric,
            max_iters=10,
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

        model_sam = SamSingleton(
            model_path="resources/sam3.pt",
            conf_thresh=0.25
        )
        labeler = SAM3AutoLabeler(model_sam, class_id=0)

        pseudo_lbl_dir = work_dir / "pseudo/labels"
        pseudo_lbl_dir.mkdir(parents=True, exist_ok=True)

        labeler.label_folder(
            image_dir=unlabeled_img_dir,
            out_label_dir=pseudo_lbl_dir,
            prompt_filter=prompt_filter,
            visualize=False
        )

        # ----------------------------------------------------
        # (5) Mixed train with samples with labels and pseudo-labels
        # ----------------------------------------------------
        self.logger.info("(5) Mixed train with samples with labels and pseudo-labels")

        mixed_img_dir = work_dir / "mixed/images"
        mixed_lbl_dir = work_dir / "mixed/labels"
        mixed_img_dir.mkdir(parents=True, exist_ok=True)
        mixed_lbl_dir.mkdir(parents=True, exist_ok=True)

        for img in labeled_img_dir.glob("*"):
            shutil.copy(img, mixed_img_dir / img.name)
            shutil.copy(
                labeled_lbl_dir / f"{img.stem}.txt",
                mixed_lbl_dir / f"{img.stem}.txt",
            )

        for img in unlabeled_img_dir.glob("*"):
            lbl = pseudo_lbl_dir / f"{img.stem}.txt"
            if lbl.exists():
                shutil.copy(img, mixed_img_dir / img.name)
                shutil.copy(lbl, mixed_lbl_dir / lbl.name)

        mixed_path = mixed_img_dir.relative_to(work_dir)
        val_path = val_img_dir.relative_to(work_dir)

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
            model_type=self.model_type,
            data_yaml=str(mixed_yaml),
            output_dir=work_dir / "yolo_train"
        )

        _ = trainer.train(**train_kwargs, project=work_dir, name="yolo_train")
        final_weights = work_dir / "yolo_train/weights/best.pt"

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

        return test_results.results_dict, work_dir


if __name__ == "__main__":

    REPEATS = 2

    dataset_name = "medical-pills"
    output_root = Path(f"runs/{dataset_name}")
    data_yaml = f"data/datasets/{dataset_name}/{dataset_name}.yaml"

    exp = ObjectDetectionExperiment(
        dataset_root=Path(f"data/datasets/{dataset_name}"),
        data_yaml=data_yaml,
        model_type="yolo11n.pt",
        output_root=output_root,
        random_state=42
    )

    train_kwargs = dict(
        epochs=100,
        imgsz=640,
        batch=16,
    )

    test_kwargs = dict(
        imgsz=640,
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

    """
    # PSEUDO WEAK YOLO
    weak_n_labeled_values = [10, 25]
    for n_labeled in weak_n_labeled_values:
        print(f"\n=== WEAK YOLO | n_labeled={n_labeled} | repeats={REPEATS} ===")
        results = exp.run(
            mode=ExperimentMode.WEAK_YOLO_PSEUDO,
            n_labeled=n_labeled,
            n_val=n_labeled,
            train_kwargs=train_kwargs,
            test_kwargs=test_kwargs,
            repeats=REPEATS
        )
    """

    # PSEUDO SAM
    sam_n_labeled_values = [3, 5]
    for n_labeled in sam_n_labeled_values:
        results = exp.run(
            mode=ExperimentMode.SAM3_PSEUDO,
            n_labeled=n_labeled,
            n_val=n_labeled,
            train_kwargs=train_kwargs,
            test_kwargs=test_kwargs,
            repeats=REPEATS,
        )
