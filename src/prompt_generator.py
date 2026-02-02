from typing import List


class PromptGenerator:
    INITIAL_PROMPT = (
        "Generate a short description (max {max_words} words) of the marked objects "
        "with red rectangles or polygons, including colour, shape, and visual appearance. "
        "Use the following images: [IMAGES]. "
        "Generate ONE concise summary description. Use singular form only."
        "Do NOT use plural nouns (e.g. 'cars', 'people')."
        "Ignore the red rectangles or polygons themselves."
    )

    ITER_PROMPT = (
        "Generate a short description (max {max_words} words) of the marked objects "
        "with red rectangles or polygons, including colour, shape, and visual appearance. "
        "Use the following images: [IMAGES]. "
        "Generate ONE concise summary description. Use singular form only."
        "Do NOT use plural nouns (e.g. 'cars', 'people')."
        "Ignore the red rectangles or polygons themselves."

        "IMPORTANT CONSTRAINTS:\n"
        "- The description MUST be meaningfully different from previous ones\n\n"

        "Previous best descriptions and their AP50 scores:\n"
        "{history}\n\n"

        "Generate a new, diverse description that explores a different visual hypothesis. "
        "Higher IoU is desirable, but diversity is REQUIRED."
    )

    SINGLE_WORD_PROMPT = (
        "Generate EXACTLY ONE WORD that best describes the marked object "
        "(e.g. object category, dominant visual concept, or material). "
        "Use the following images: [IMAGES]. "
        "Do NOT use articles, adjectives chains, or punctuation. "
        "Return a single lowercase word only. Use singular form only."
        "Do NOT use plural nouns (e.g. 'cars', 'people')."
    )

    SINGLE_WORD_ITER_PROMPT = (
        "Generate EXACTLY ONE WORD that best describes the marked object "
        "(e.g. object category, dominant visual concept, or material). "
        "Use the following images: [IMAGES]. "
        "Do NOT use articles, adjectives chains, or punctuation. "
        "Return a single lowercase word only."

        "Previous words and their AP50 scores:\n"
        "{history}\n\n"

        "Rules:\n"
        "- Use a single lowercase word\n"
        "- Do NOT repeat previous words\n"
        "- Use singular form only\n"
        "- Do NOT use plural nouns (e.g. 'cars', 'people')"
    )

    @staticmethod
    def build_initial(images: List[str], max_words: int) -> str:
        if max_words == 1:
            return (
                PromptGenerator.SINGLE_WORD_PROMPT
                .replace("[IMAGES]", ", ".join(images))
            )

        return (
            PromptGenerator.INITIAL_PROMPT
            .format(max_words=max_words)
            .replace("[IMAGES]", ", ".join(images))
        )

    @staticmethod
    def build_iterative(images: List[str], history: List[dict], max_words: int) -> str:

        hist_str = "\n".join(
            f"[{i + 1}] {h['desc']} → AP50={h['iou']:.4f}"
            for i, h in enumerate(history)
        )

        if max_words == 1:
            return (
                PromptGenerator.SINGLE_WORD_ITER_PROMPT
                .format(history=hist_str)
                .replace("[IMAGES]", ", ".join(images))
            )

        return (
            PromptGenerator.ITER_PROMPT
            .format(
                max_words=max_words,
                history=hist_str,
            )
            .replace("[IMAGES]", ", ".join(images))
        )
