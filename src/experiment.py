import random
import shutil
import yaml
import numpy as np

from pathlib import Path
from enum import Enum
from typing import Dict

from ultralytics import YOLO

from src.utils import DetectionAP50
from src.dataset_layout import DatasetLayout
from src.experiment_logger import ExperimentLogger
from src.yolo_subset_sampler import YoloSubsetSampler
from src.yolo_train_eval import YOLOTrainer, YOLOEvaluator
from src.yolo_weak_auto_labeler import WeakYoloSingleton, YOLOWeakAutoLabeler


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
        self.sampler = YoloSubsetSampler(random_state)
        self.metric = DetectionAP50()

    def run(self, mode: ExperimentMode, **kwargs):
        self.logger.info(f"Running mode: {mode.value}")

        if mode == ExperimentMode.STANDARD:
            return self._run_standard(**kwargs)

        if mode == ExperimentMode.WEAK_YOLO_PSEUDO:
            return self._run_weak_yolo_pseudo(**kwargs)

        if mode == ExperimentMode.SAM3_PSEUDO:
            return self._run_sam3_pseudo(**kwargs)

        raise ValueError(f"Unsupported mode: {mode}")

    def _run_standard(self, train_kwargs: dict, test_kwargs: dict):
        self.logger.info("STANDARD: training on full dataset")

        work_dir = self.output_root / "standard_yolo"
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

        return test_results.results_dict

    def _run_weak_yolo_pseudo(
        self,
        n_labeled: int,
        n_val: int,
        train_kwargs: Dict,
        test_kwargs: Dict,
        pseudo_kwargs: Dict,
    ):
        self.logger.info("WEAK YOLO PSEUDO-LABELING")

        work_dir = self.output_root / "weak_yolo"
        work_dir.mkdir(parents=True, exist_ok=True)

        # ----------------------------------------------------
        # (1) Prepare sets of labeled and unlabeled samples for training
        # ----------------------------------------------------
        self.logger.info("(1) Prepare sets of labeled and unlabeled samples for training")

        train_imgs = sorted(self.dataset.images("train").glob("*.*"))
        assert n_labeled < len(train_imgs)

        labeled_imgs = set(random.sample(train_imgs, n_labeled))
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
        val_sample = set(random.sample(val_imgs, n_val))

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
        return test_results.results_dict

    def _run_sam3_pseudo(self):
        pass


if __name__ == "__main__":

    dataset_name = "crack-seg"
    output_root = Path(f"runs/{dataset_name}")
    data_yaml = f"data/datasets/{dataset_name}/{dataset_name}.yaml"

    exp = ObjectDetectionExperiment(
        dataset_root=Path(f"data/datasets/{dataset_name}"),
        data_yaml=data_yaml,
        model_type="yolo11n.pt",
        output_root=output_root
    )

    train_kwargs = dict(
        epochs=100,
        imgsz=640,
        batch=16,
    )

    test_kwargs = dict(
        imgsz=640,
    )

    pseudo_kwargs = dict(
        conf=0.6,
    )

    """
    # STANDARD
    results = exp.run(
        mode=ExperimentMode.STANDARD,
        train_kwargs=train_kwargs,
        test_kwargs=test_kwargs
    )
    """

    # WEAK_YOLO_PSEUDO
    results = exp.run(
        mode=ExperimentMode.WEAK_YOLO_PSEUDO,
        n_labeled=100,
        n_val=100,
        train_kwargs=train_kwargs,
        test_kwargs=test_kwargs,
        pseudo_kwargs=pseudo_kwargs
    )
