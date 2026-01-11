from ollama import chat
from typing import List


class OllamaVisionLLM:
    def __init__(self, model: str):
        self.model = model
        self.prompt_template = (
            "Generate short description (max 10 words) of marked objects "
            "with red rectangles including information about colour, shape "
            "and characteristic patterns using following images: [IMAGES]. "
            "Remember to generate a single summary description, not separate descriptions for each object in each image."
            "Example of possible generated description: 'elongated dark brown shape with visible spots'."
        )

    def inference(self, image_paths: List[str]):
        images_str = ", ".join(image_paths)
        prompt = self.prompt_template.replace("[IMAGES]", images_str)
        messages = [{
            "role": "user",
            "content": prompt,
            "images": image_paths
        }]
        response = chat(model=self.model, messages=messages)
        return response["message"]["content"]


if __name__ == "__main__":

    for model_id, model_name in enumerate(["qwen3-vl:latest", "gemma3:latest", "ministral-3:latest"]):

        model = OllamaVisionLLM(model_name)
        path_to_images = "data/example_images/"

        img_names = ["insect_example_3.png", "insect_example_4.jpg"]
        image_paths = [path_to_images + img_name for img_name in img_names]

        description = model.inference(image_paths)
        print(f"model_id={model_id}")
        print(f"description={description}")
        print("------------")
