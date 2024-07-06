from yaml import safe_load


class EvalConfig:
    def __init__(self, config_path="./eval.yml"):
        with open(config_path) as file_path:
            self.config = safe_load(file_path)

        self.eval_name = self.config["eval_name"]
        self.seed = self.config["seed"]
        self.eval_amount = self.config["eval_amount"]
        self.bad_data_handle = self.config["bad_data"]
        self.test_group = []
    
    def get_eval_name(self):
        return self.eval_name
    
    def get_seed(self):
        return self.seed

    def get_test_group(self):
        test_group = [test for test in self.config["test_group"].values()]

        return test_group
