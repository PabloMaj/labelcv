from pathlib import Path


class DatasetLayout:
    """
    Assumed structure:
    root/
      images/{train,val,test}
      labels/{train,val,test}
    """

    def __init__(self, root: Path):
        self.root = root
        self.images_root = root / "images"
        self.labels_root = root / "labels"

    def images(self, split: str) -> Path:
        return self.images_root / split

    def labels(self, split: str) -> Path:
        return self.labels_root / split

    def assert_valid(self):
        for split in ("train", "val", "test"):
            assert self.images(split).exists(), f"Missing images/{split}"
            assert self.labels(split).exists(), f"Missing labels/{split}"


if __name__ == "__main__":
    dataset_root = Path("data/datasets/medical-pills")
    ds = DatasetLayout(dataset_root)

    print("Train images:", ds.images("train"))
    print("Val labels:", ds.labels("val"))
    print("Test images:", ds.images("test"))
