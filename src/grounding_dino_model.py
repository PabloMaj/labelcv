import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from accelerate import Accelerator


class GroundingDinoSingleton:
    _instance = None
    _model = None
    _device = Accelerator().device
    _model_id = "IDEA-Research/grounding-dino-tiny"

    def __new__(cls):

        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._model = AutoModelForZeroShotObjectDetection.from_pretrained(cls._model_id).to(cls._device)
        return cls._instance

    def predict(self, path_to_image: str = None, prompt_filter: str = None, conf_thresh: float = 0.25):
        processor = AutoProcessor.from_pretrained(self._model_id)
        image = Image.open(path_to_image)
        inputs = processor(images=image, text=prompt_filter, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            outputs = self._model(**inputs)
        results = processor.post_process_grounded_object_detection(outputs, inputs.input_ids, threshold=conf_thresh, text_threshold=0.4, target_sizes=[image.size[::-1]])
        return results


if __name__ == "__main__":

    path_to_image = "C:/projects/labelcv/data/datasets/african-wildlife_zebra/images/test/4 (274).jpg"
    prompt_filter = [["stripped pattern"]]

    model_grounding_dino = GroundingDinoSingleton()
    results = model_grounding_dino.predict(path_to_image, prompt_filter, 0.25)
    print(results)
