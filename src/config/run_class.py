from yaml import safe_load


class RunConfig:
    def __init__(self, config_path="./run.yml"):
        with open(config_path) as file_path:
            self.config = safe_load(file_path)

        self.test_name = self.config["test_name"]
        self.seed = self.config["seed"]
        self.__dataset_cfg = self.config["dataset"]
        self.__model_cfg = self.config["model"]
        self.__prompt_cfg = self.config["prompt"]
        self.__run_params = self.config["run_params"]

    def get_test_name(self):
        return self.test_name

    def get_seed(self):
        return self.seed

    def get_model_config(self):
        model_cfg = {
            "name": self.__model_cfg["name"],
            "path": self.__model_cfg["path"],
            "family": self.__model_cfg["family"],
            "use_8_bit": int(self.__model_cfg["params"]["use_8_bit"]),
            "device": self.__model_cfg["params"]["device"],
            "low_cpu": int(self.__model_cfg["params"]["low_cpu"]),
        }

        return model_cfg

    def get_data_config(self):
        data_cfg = {
            "name": self.__dataset_cfg["name"],
            "use_scene_graph": self.__dataset_cfg["use_scene_graph"],
            "count": int(self.__dataset_cfg["count"]),
        }

        return data_cfg

    def get_prompt(self):
        raw_path = self.__prompt_cfg["prompt"]

        folder = raw_path.split("-")[0]
        filename = raw_path.split("-")[1]

        prompt_path = f"./prompt/{folder}/{filename}.txt"

        with open(prompt_path) as file:
            prompt = file.read()

        return prompt

    def get_run_params(self):
        run_params = {
            "num_per_inference": int(self.__run_params["num_per_inference"]),
            "use_img_ext": self.__run_params["use_img_ext"],
            "q_prefix": eval(self.__run_params["q_prefix"]),
            "q_prefix_prop": eval(self.__run_params["q_prefix_prop"]),
        }

        return run_params
