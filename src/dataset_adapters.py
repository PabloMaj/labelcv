from pathlib import Path
import shutil
import yaml
import os
import random


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
                        if len(line) > 1 and int(line.split()[0]) == class_id:
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


class MultiSubsetYoloAdapter:
    def __init__(
        self,
        dataset_root: str,
        allowed_subsets: list[str] | None = None,
        output_root: str = "yolo_dataset",
        split=(0.6, 0.2, 0.2),
        class_name: str = "object",
        random_state: int | None = None,
        prefix: str = ""
    ):
        self.dataset_root = Path(dataset_root)
        self.allowed_subsets = allowed_subsets
        self.output_root = Path(output_root)
        self.split = split
        self.class_name = class_name
        self.random_state = random_state
        self.prefix = prefix

        self.images_root = self.dataset_root / "images"
        self.labels_root = self.dataset_root / "labels"

        # jeśli allowed_subsets nie podano, bierz wszystkie podfoldery
        if self.allowed_subsets is None:
            self.allowed_subsets = [
                f.name for f in self.images_root.iterdir() if f.is_dir()
            ]

    def create_datasets(self):
        for subset in self.allowed_subsets:
            self._create_single_subset_dataset(subset)

    def _create_single_subset_dataset(self, subset: str):
        img_dir = self.images_root / subset
        lbl_dir = self.labels_root / subset

        if not img_dir.exists() or not lbl_dir.exists():
            print(f"Warning: {subset} not found, skipping")
            return

        # zbierz obrazy i labeli
        all_images = []
        all_labels = []
        for img_file in img_dir.glob("*.*"):
            label_file = lbl_dir / f"{img_file.stem}.txt"
            if label_file.exists():
                all_images.append(img_file)
                all_labels.append(label_file)

        # mieszaj
        combined = list(zip(all_images, all_labels))
        if self.random_state is not None:
            random.seed(self.random_state)
        random.shuffle(combined)
        all_images, all_labels = zip(*combined) if combined else ([], [])

        # split
        n = len(all_images)
        n_train = int(n * self.split[0])
        n_val = int(n * self.split[1])

        splits = {
            "train": (all_images[:n_train], all_labels[:n_train]),
            "val": (all_images[n_train:n_train + n_val], all_labels[n_train:n_train + n_val]),
            "test": (all_images[n_train + n_val:], all_labels[n_train + n_val:])
        }

        # folder docelowy z prefixem
        out_dir = self.output_root / f"{self.prefix}{subset}"

        for split_name, (imgs, lbls) in splits.items():
            out_img_dir = out_dir / "images" / split_name
            out_lbl_dir = out_dir / "labels" / split_name
            out_img_dir.mkdir(parents=True, exist_ok=True)
            out_lbl_dir.mkdir(parents=True, exist_ok=True)

            for img_file, lbl_file in zip(imgs, lbls):
                shutil.copy(img_file, out_img_dir / img_file.name)

                with open(lbl_file, encoding="utf-8") as f:
                    lines = f.readlines()

                lines_remap = []
                for line in lines:
                    parts = line.strip().split()
                    parts[0] = "0"  # jednoklasowy
                    lines_remap.append(" ".join(parts) + "\n")

                with open(out_lbl_dir / lbl_file.name, "w", encoding="utf-8") as f:
                    f.writelines(lines_remap)

        # YAML
        yaml_data = {
            "path": str(out_dir.resolve()),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": {0: self.class_name}
        }

        with open(out_dir / "data.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(yaml_data, f, allow_unicode=True)


if __name__ == "__main__":

    # datasets with many classes (construction-ppe, african-wildlife)
    allowed_classes_per_dataset = {
        "construction-ppe": ["helmet", "gloves", "vest", "goggles"],
        "african-wildlife": ["buffalo", "elephant", "rhino", "zebra"],
        "homeobjects": ["chair", "table", "lamp", "potted plant"]

    }

    path_to_datasets = "data/datasets/"

    for dataset_name in ["construction-ppe", "african-wildlife", "homeobjects"]:

        yaml_file_name = [el for el in os.listdir(path_to_datasets + dataset_name) if "yaml" in el][0]

        filter = YoloDatasetFilter(
            yaml_path=path_to_datasets + f"{dataset_name}/{yaml_file_name}",
            allowed_classes=allowed_classes_per_dataset[dataset_name],
            output_root=path_to_datasets
        )

        filter.create_filtered_datasets()

    # datasets with one class but no yolo structure (global-wheat)
    adapter = MultiSubsetYoloAdapter(
        dataset_root="data/datasets/global-wheat-codalab-official/",
        allowed_subsets=["arvalis_1", "arvalis_3", "ethz_1", "rres_1"],
        output_root="data/datasets/",
        split=(0.6, 0.2, 0.2),
        class_name="spike",
        random_state=42,
        prefix="wheat_"
    )

    adapter.create_datasets()
