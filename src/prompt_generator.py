from typing import List


class PromptGenerator:
    INITIAL_PROMPT = (
        "Generate a short description (max 10 words) of the marked objects "
        "with red rectangles, including colour, shape, and visual appearance. "
        "Use the following images: [IMAGES]. "
        "Generate ONE concise summary description. "
        "Ignore the red rectangles themselves."
    )

    ITER_PROMPT = (
        "Generate a short description (max 10 words) of the marked objects "
        "with red rectangles, including colour, shape, and visual appearance. "
        "Use the following images: [IMAGES]. "
        "Generate ONE concise summary description. "
        "Ignore the red rectangles themselves."

        "IMPORTANT CONSTRAINTS:\n"
        "- The description MUST be meaningfully different from previous ones\n\n"

        "Previous best descriptions and their AP50 scores:\n"
        "{history}\n\n"

        "Generate a new, diverse description that explores a different visual hypothesis. "
        "Higher IoU is desirable, but diversity is REQUIRED."
    )

    @staticmethod
    def build_initial(images: List[str]) -> str:
        return PromptGenerator.INITIAL_PROMPT.replace(
            "[IMAGES]", ", ".join(images)
        )

    @staticmethod
    def build_iterative(images: List[str], history: List[dict]) -> str:

        hist_str = "\n".join(
            f"[{i + 1}] {h['desc']} → AP50={h['iou']:.4f}"
            for i, h in enumerate(history)
        )

        return (
            PromptGenerator.ITER_PROMPT
            .format(
                history=hist_str
            )
            .replace("[IMAGES]", ", ".join(images))
        )
