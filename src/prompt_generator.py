from typing import List


class PromptGenerator:
    INITIAL_PROMPT = (
        "Generate short description (max 10 words) of marked objects "
        "with red rectangles including information about colour, shape "
        "and characteristic patterns using following images: [IMAGES]. "
        "Remember to generate a single summary description, not separate descriptions "
        "for each object in each image. Ignore the red bounding boxes in the description—they "
        "are included solely to indicate object locations."
        "Example of possible generated description: "
        "'elongated dark brown shape with visible spots'."
    )

    ITER_PROMPT = (
        "Your task is to generate short description (max 10 words) of marked objects "
        "with red rectangles including information about colour, shape and characteristic "
        "patterns using following images: [IMAGES]. "
        "Remember to generate a single summary description, not separate descriptions "
        "for each object in each image. Ignore the red bounding boxes in the description—they "
        "are included solely to indicate object locations."
        "Example of possible generated description: "
        "'elongated dark brown shape with visible spots'.\n\n"
        "Take into account that the generated descriptions are improved in subsequent "
        "iterations. In previous iterations, the descriptions were as follows, "
        "along with the assigned computed AP50det between the label proposals "
        "and the ground truth:\n\n"
        "{history}\n\n"
        "Considering the results from the previous iterations, generate a better "
        "description (corresponding to a higher IoU). Try to generate a significantly different description "
        "each time to capture the most relevant and key words."
    )

    @staticmethod
    def build_initial(images: List[str]) -> str:
        return PromptGenerator.INITIAL_PROMPT.replace(
            "[IMAGES]", ", ".join(images)
        )

    @staticmethod
    def build_iterative(images: List[str], history: List[dict]) -> str:
        hist_str = "\n".join(
            f"[{i + 1}] Description: {h['desc']}. IoU: {h['iou']:.4f}"
            for i, h in enumerate(history)
        )
        return PromptGenerator.ITER_PROMPT.format(history=hist_str).replace(
            "[IMAGES]", ", ".join(images)
        )
