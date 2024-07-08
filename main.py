from src.config.run import RunConfig
from src.dataset import Dataset
from src.model import LVLM
from src.runner.generate import GenerateRunner
from src.helper.parser import export_result

from time import time

import pprint


if __name__ == "__main__":
    t_start = time()

    Config = RunConfig()
    DataConfig = Config.get_data_config()
    RunParamsConfig = Config.get_run_params()

    dataset = Dataset(DataConfig, RunParamsConfig)
    model = LVLM()
    prompt = Config.get_prompt()

    runner = GenerateRunner(dataset, model, prompt)
    synthetic_data, raw_data = runner.generate_synthetic_data()

    t_run = time() - t_start

    export_result(synthetic_data, raw_data, t_run, Config)
