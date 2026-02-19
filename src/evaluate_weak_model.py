import torch
import gc
import os
from pathlib import Path

from src.yolo_train_eval import YOLOEvaluator


def cuda_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


if __name__ == "__main__":

    test_kwargs = dict(imgsz=640)

    path_to_read_results = "runs/"

    for dataset_name in os.listdir(path_to_read_results):
        for num_folder in os.listdir(path_to_read_results + f"{dataset_name}/weak_yolo/"):
            for repeat_folder in os.listdir(path_to_read_results + f"{dataset_name}/weak_yolo/{num_folder}/"):

                data_yaml = f"data/datasets/{dataset_name}/data.yaml"
                weights_path = Path(path_to_read_results + f"{dataset_name}/weak_yolo/{num_folder}/{repeat_folder}/stage1/weights/best.pt")
                work_dir = Path(path_to_read_results + f"{dataset_name}/only_labeled/{num_folder}/{repeat_folder}/")

                evaluator = YOLOEvaluator(
                    weights_path=weights_path,
                    data_yaml=str(data_yaml),
                    output_dir=work_dir / "eval_test",
                )

                test_results = evaluator.test(**test_kwargs, project=work_dir, name="eval_test")

                del evaluator
                cuda_cleanup()
