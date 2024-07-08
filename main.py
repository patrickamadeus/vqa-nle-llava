from src.config.run_class import RunConfig
from src.dataset import Dataset
from src.model import LVLM
from src.runner.generate import GenerateRunner


if __name__ == "__main__":
    Config = RunConfig()
    DataConfig = Config.get_data_config()
    RunParamsConfig = Config.get_run_params()

    dataset = Dataset(DataConfig, RunParamsConfig)
    model = LVLM()
    prompt = Config.get_prompt()

    runner = GenerateRunner(dataset, model, prompt, ensemble=True)
    synthetic_data, raw_data = runner.generate_synthetic_data()

    print(len(synthetic_data))
