import random
import shutil
from pathlib import Path
from typing import List, Tuple


class YoloSubsetSampler:
    def __init__(
        self,
        random_state: int,
        img_exts=(".jpg", ".jpeg", ".png"),
    ):
        self.random_state = random_state
        self.img_exts = img_exts

    def sample(
        self,
        images_dir: Path,
        labels_dir: Path,
        out_images_dir: Path,
        out_labels_dir: Path,
        n_samples: int,
    ) -> Tuple[List[Path], List[Path]]:

        images = []
        for ext in self.img_exts:
            images.extend(images_dir.glob(f"*{ext}"))

        images = sorted(images)
        rng = random.Random(self.random_state)
        selected = rng.sample(images, min(n_samples, len(images)))

        out_images_dir.mkdir(parents=True, exist_ok=True)
        out_labels_dir.mkdir(parents=True, exist_ok=True)

        out_imgs, out_lbls = [], []

        for img in selected:
            lbl = labels_dir / f"{img.stem}.txt"
            assert lbl.exists(), f"Missing label for {img.name}"

            shutil.copy(img, out_images_dir / img.name)
            shutil.copy(lbl, out_labels_dir / lbl.name)

            out_imgs.append(out_images_dir / img.name)
            out_lbls.append(out_labels_dir / lbl.name)

        return out_imgs, out_lbls


if __name__ == "__main__":
    sampler = YoloSubsetSampler(random_state=123)

    # Ścieżki przykładowe – dostosuj do realnego datasetu
    images = Path("data/datasets/medical-pills/images/train")
    labels = Path("data/datasets/medical-pills/labels/train")

    out_imgs = Path("runs/sampler_test/images/train")
    out_lbls = Path("runs/sampler_test/labels/train")

    sampler.sample(images, labels, out_imgs, out_lbls, n_samples=5)
