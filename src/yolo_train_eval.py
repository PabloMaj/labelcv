import json
from pathlib import Path
from ultralytics import YOLO


class YOLOTrainer:
    """
    Class responsible for training a YOLO model.
    """
    def __init__(self, model_type: str, data_yaml: str, output_dir: Path):
        self.model_type = model_type
        self.data_yaml = data_yaml
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model = YOLO(f"resources/{model_type}")

    def train(self, **train_kwargs):
        """
        Run training and save training history and best validation metrics.
        """
        results = self.model.train(data=self.data_yaml, **train_kwargs)

        history_path = self.output_dir / "train_history.json"
        best_val_path = self.output_dir / "best_val_metrics.json"

        # Training history
        history = results.results_dict if hasattr(results, "results_dict") else {}
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        # Best validation metrics
        best_metrics = {}
        if hasattr(results, "best"):
            best_metrics = results.best

        with open(best_val_path, "w", encoding="utf-8") as f:
            json.dump(best_metrics, f, indent=2)

        return results


class YOLOEvaluator:
    """
    Class responsible for evaluating a YOLO model on the test split.
    """
    def __init__(self, weights_path: str, data_yaml: str, output_dir: Path):
        self.weights_path = weights_path
        self.data_yaml = data_yaml
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model = YOLO(weights_path)

    def test(self, **test_kwargs):
        """
        Run evaluation on the test set and save metrics.
        """
        results = self.model.val(data=self.data_yaml, split="test", **test_kwargs)

        test_metrics_path = self.output_dir / "test_metrics.json"
        metrics = results.results_dict if hasattr(results, "results_dict") else {}

        with open(test_metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        return results


def main():
    """
    Example code-based execution (no command-line arguments).
    Adjust paths and parameters to your project.
    """

    # ---------------- CONFIGURATION ----------------
    MODE = "test"  # "train" or "test"

    data_yaml = "data/datasets/crack-seg/crack-seg.yaml"
    output_dir = Path("runs/yolo_train_eval")

    if MODE == "train":
        model_type = "yolov8m.pt"

        trainer = YOLOTrainer(
            model_type=model_type,
            data_yaml=data_yaml,
            output_dir=output_dir,
        )

        trainer.train(
            epochs=30,
            imgsz=416,
            batch=16,
        )

    elif MODE == "test":
        weights_path = "runs/detect/train/weights/best.pt"

        evaluator = YOLOEvaluator(
            weights_path=weights_path,
            data_yaml=data_yaml,
            output_dir=output_dir / "test",
        )

        evaluator.test(imgsz=416)


if __name__ == "__main__":
    main()
