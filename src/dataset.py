import json
import os
from PIL import Image
import torch
import torchvision
import random

from src.config.run_class import RunConfig
from src.base import set_seed

Config = RunConfig()
DataConfig = Config.get_data_config()
RunParamsConfig = Config.get_run_params()
SEED = Config.get_seed()

set_seed(SEED)


class Dataset:
    def __init__(self, data_config=DataConfig, run_params_config=RunParamsConfig):
        self.__img_path = "./dataset/img"
        self.__use_scene_graph = data_config["use_scene_graph"]
        self.__count = data_config["count"]
        self.__q_prefix_choice = run_params_config["q_prefix"]
        self.__q_prefix_prop = run_params_config["q_prefix_prop"]
        self.__num_per_inference = run_params_config["num_per_inference"]

        if self.__use_scene_graph:
            with open("./dataset/scene_graph.json") as file:
                self.__scene_graph = json.load(file)

        self.__images = []
        self.__object_names = []
        self.__q_prefices = []

    def __get_filename(long_path: str) -> str:
        filename_with_ext = os.path.basename(long_path)
        filename = os.path.splitext(filename_with_ext)[0]

        return filename, filename_with_ext

    def __get_imagepaths_with_annotation_info(self):
        keys = set(self.__scene_graph.keys())

        file_paths = [
            os.path.join(root, file)
            for root, _, files in os.walk(self.__img_path)
            for file in files
            if (os.path.basename(root) == os.path.basename(self.__img_path))
            and (self.__get_filename(file)[-1] in keys)
        ][: self.__count]

        return file_paths

    def __annotate_image(self, img_torch, bbox):
        bbox = torch.tensor([bbox])

        img_tensor = torchvision.utils.draw_bounding_boxes(
            img_torch, bbox, width=3, colors=["red"]
        )
        img_pil = torchvision.transforms.ToPILImage()(img_tensor)

        return img_pil

    def __multi_annotate_image(self, img_path, num_obj=5, min_area_div=100):
        img_torch = torchvision.io.read_image(img_path)

        img_area = img_torch.size()[-1] * img_torch.size()[-2]
        graph = self.__scene_graph[self.__get_filename(img_path)[-1]]

        annotated_imgs = []
        obj_names = []

        for object in graph["objects"].values():
            x, y, w, h = object["x"], object["y"], object["w"], object["h"]
            if w * h * min_area_div < img_area:
                continue

            bbox = [x, y, x + w, y + h]
            img_pil = self.__annotate_image(img_torch, bbox)

            annotated_imgs.append(img_pil)
            obj_names.append(object["name"])

        return annotated_imgs, obj_names

    def __expand_prefix_stratify(self) -> None:
        if self.__use_scene_graph:
            prefixes = (
                "what",
                "is/am/are (pick one that fits the most)",
                "which",
                "how many",
                "where/when (pick one that fits the most)",
            )
            prefixes_proportions = [2, 2, 2, 2, 1, 1, 1]
        else:
            prefixes = self.__q_prefix_choice
            prefixes_proportions = self.__q_prefix_prop

        for i, prefix in enumerate(prefixes):
            self.__q_prefices.extend([prefix] * prefixes_proportions[i])

        random.shuffle(self.__q_prefices)

    def get_images_and_obj_names(self):
        img_paths = self.__get_imagepaths_with_annotation_info()

        for img_path in img_paths:
            if self.__use_scene_graph:
                annotated_imgs, obj_names = self.__multi_annotate_image(img_path)
                self.__images.extend(annotated_imgs)
                self.__object_names.extend(obj_names)
            else:
                img = Image.open(img_path)
                self.__images.append(img)

        return self.__images, self.__object_names

    def get_data(self):
        self.__expand_prefix_stratify()
        images, object_names = self.get_images_and_obj_names()

        return images, self.__q_prefices, object_names
