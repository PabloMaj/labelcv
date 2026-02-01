from enum import Enum
from pathlib import Path


class TaskType(Enum):
    DETECT = "detect"
    SEGMENT = "segment"


def detect_task_from_labels(labels_dir: Path) -> TaskType:
    for lbl in labels_dir.glob("*.txt"):
        with open(lbl, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) > 5:
                    return TaskType.SEGMENT
                elif len(parts) == 5:
                    return TaskType.DETECT
    raise RuntimeError(
        f"Cannot determine task type from labels in {labels_dir}"
    )
