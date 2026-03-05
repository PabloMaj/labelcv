from src.sam_model import SamSingleton
from src.grounding_dino_model import GroundingDinoSingleton


class ResultsPostProcessor:
    def process(self, results=None, model_type: str = None):
        if model_type == "sam":
            boxes_coords = []
            conf_values = []
            boxes_all = results[0].boxes
            for box in boxes_all:
                boxes_coords.append(box.xyxy.cpu().numpy().tolist()[0])
                conf_values.append(box.conf.cpu().numpy().tolist()[0])
        elif model_type == "grounding_dino":
            boxes_coords = results[0]["boxes"].cpu().numpy().tolist()
            conf_values = results[0]["scores"].cpu().numpy().tolist()
        return boxes_coords, conf_values


if __name__ == "__main__":

    # settings
    model_path_sam = "resources/sam3.pt"
    conf_thresh = 0.25
    save_flag = False
    path_to_image = "C:/projects/labelcv/data/datasets/crack-seg/images/test/2323.rf.06768caef93b3a44d3c59325a03230eb.jpg"
    prompt_filter = "crack"

    # initialization SAM and Grounding Dino
    model_sam = SamSingleton(model_path=model_path_sam, conf_thresh=conf_thresh, save_flag=save_flag)
    model_grounding_dino = GroundingDinoSingleton()

    # inference SAM and Grounding Dino
    results_post_processor = ResultsPostProcessor()
    results_sam = model_sam.predict(path_to_image=path_to_image, prompt_filter=prompt_filter)
    boxes_coords, conf_values = results_post_processor.process(results_sam, "sam")
    print(f"boxes_coords={boxes_coords}\tconf_values={conf_values}")

    prompt_filter_formatted = [[prompt_filter]]
    results_ground_dino = model_grounding_dino.predict(path_to_image=path_to_image, prompt_filter=prompt_filter_formatted, conf_thresh=conf_thresh)
    boxes_coords, conf_values = results_post_processor.process(results_ground_dino, "grounding_dino")
    print(f"boxes_coords={boxes_coords}\tconf_values={conf_values}")
