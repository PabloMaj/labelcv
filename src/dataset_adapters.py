from pathlib import Path
import shutil
import yaml
import os


class YoloDatasetFilter:
    def __init__(
        self,
        yaml_path: str,
        allowed_classes: list[str],
        output_root: str
    ):
        self.yaml_path = Path(yaml_path)
        self.dataset_root = self.yaml_path.parent
        self.dataset_name = self.dataset_root.name
        self.allowed_classes = allowed_classes
        self.output_root = Path(output_root)

        with open(self.yaml_path, "r", encoding="utf-8") as f:
            self.data = yaml.safe_load(f)

        self.class_name_to_id = {
            name: idx for idx, name in self.data["names"].items()
        }

    def create_filtered_datasets(self):
        for class_name in self.allowed_classes:
            if class_name not in self.class_name_to_id:
                continue

            class_id = self.class_name_to_id[class_name]
            out_dir = self.output_root / f"{self.dataset_name}_{class_name}"

            self._create_single_class_dataset(
                class_name=class_name,
                class_id=class_id,
                output_dir=out_dir
            )

    def _create_single_class_dataset(
        self,
        class_name: str,
        class_id: int,
        output_dir: Path
    ):
        for split in ["train", "val", "test"]:
            if split not in self.data:
                continue

            img_dir = self.dataset_root / self.data[split]
            lbl_dir = Path(str(img_dir).replace("images", "labels"))

            out_img_dir = output_dir / "images" / split
            out_lbl_dir = output_dir / "labels" / split

            out_img_dir.mkdir(parents=True, exist_ok=True)
            out_lbl_dir.mkdir(parents=True, exist_ok=True)

            for label_file in lbl_dir.glob("*.txt"):
                kept_lines = []

                with open(label_file) as f:
                    for line in f:
                        if int(line.split()[0]) == class_id:
                            # remap ID -> 0
                            parts = line.split()
                            parts[0] = "0"
                            kept_lines.append(" ".join(parts) + "\n")

                if not kept_lines:
                    continue

                image_file = img_dir / (label_file.stem + ".jpg")
                if not image_file.exists():
                    continue

                shutil.copy(image_file, out_img_dir / image_file.name)

                with open(out_lbl_dir / label_file.name, "w") as f:
                    f.writelines(kept_lines)

        self._save_yaml(output_dir, class_name)

    def _save_yaml(self, output_dir: Path, class_name: str):
        new_yaml = {
            "path": str(output_dir.resolve()),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": {0: class_name}
        }

        with open(output_dir / "data.yaml", "w") as f:
            yaml.safe_dump(new_yaml, f)


if __name__ == "__main__":

    # datasets with many classes
    allowed_classes_per_dataset = {
        "construction-ppe": ["helmet", "gloves", "vest", "goggles"],
        "african-wildlife": ["buffalo", "elephant", "rhino", "zebra"]
    }

    path_to_datasets = "data/datasets/"

    for dataset_name in ["construction-ppe", "african-wildlife"]:

        yaml_file_name = [el for el in os.listdir(path_to_datasets + dataset_name) if "yaml" in el][0]

        filter = YoloDatasetFilter(
            yaml_path=path_to_datasets + f"{dataset_name}/{yaml_file_name}",
            allowed_classes=allowed_classes_per_dataset[dataset_name],
            output_root=path_to_datasets
        )

        filter.create_filtered_datasets()
