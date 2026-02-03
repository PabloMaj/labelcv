from pathlib import Path
from typing import Dict, List, Set


class YOLODatasetExplorer:
    def __init__(self, dataset_path: str, task: str = "detection"):
        self.dataset_path = Path(dataset_path)
        assert task in ["detection", "segmentation"], "Task must be 'detection' or 'segmentation'"
        self.task = task
        self.splits = ["train", "val", "test"]
        self.summary: Dict[str, Dict[str, int]] = {}

    def _get_files(self, split: str) -> List[Path]:
        img_dir = self.dataset_path / "images" / split
        label_dir = self.dataset_path / "labels" / split
        if not img_dir.exists() or not label_dir.exists():
            return []
        img_files = sorted(img_dir.glob("*.*"))
        return img_files

    def _count_objects(self, label_file: Path) -> int:
        if not label_file.exists():
            return 0
        with open(label_file, "r") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        return len(lines)

    def explore(self) -> Dict[str, Dict[str, int]]:
        all_file_sets: Dict[str, Set[str]] = {}

        for split in self.splits:
            img_files = self._get_files(split)
            if not img_files:
                continue

            total_objects = 0
            filenames_set = set()

            for img_file in img_files:
                label_file = self.dataset_path / "labels" / split / f"{img_file.stem}.txt"
                total_objects += self._count_objects(label_file)
                filenames_set.add(img_file.stem)

            self.summary[split] = {
                "samples": len(img_files),
                "objects": total_objects
            }
            all_file_sets[split] = filenames_set

        leakage_report = self._check_data_leakage(all_file_sets)
        self.summary["data_leakage"] = leakage_report

        return self.summary

    def _check_data_leakage(self, file_sets: Dict[str, Set[str]]) -> Dict[str, List[str]]:
        leakage = {}
        for i, split1 in enumerate(self.splits):
            for split2 in self.splits[i + 1:]:
                if split1 in file_sets and split2 in file_sets:
                    intersection = file_sets[split1] & file_sets[split2]
                    if intersection:
                        leakage[f"{split1} & {split2}"] = list(intersection)
        return leakage


if __name__ == "__main__":

    path_to_datasets = "data/datasets/"
    dataset_names = ["african-wildlife_buffalo", "african-wildlife_elephant", "african-wildlife_rhino", "african-wildlife_zebra",
                     "construction-ppe_gloves", "construction-ppe_goggles", "construction-ppe_helmet", "construction-ppe_vest",
                     "crack-seg", "medical-pills", "wheat_arvalis_1", "wheat_arvalis_3", "wheat_ethz_1", "wheat_rres_1"]
    for dataset_name in dataset_names:
        print(dataset_name)
        dataset_path = path_to_datasets + dataset_name
        explorer = YOLODatasetExplorer(dataset_path)
        summary = explorer.explore()
        print(summary)
        print("------------------------")
