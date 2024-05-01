from src.base import load_config, unpack_json

# Load from YAML
# Load from dictionary
eval_cfg = load_config("./", "eval.yml")

SEED = eval_cfg["seed"]
TEST_NAME = eval_cfg["test_name"]
TEST_GROUP = [v for v in eval_cfg["test_group"].values()]
MULTI_RESULT_PATH = f"./result/{TEST_NAME}/"
EVAL_NUM = eval_cfg["eval_amount"]

# METADATA_JSON = unpack_json(f"./result/{TEST_NAME}/metadata.json")
# DATASET_NAME = METADATA_JSON["dataset_name"]

# DATASET_IMG_PATH = f"./dataset/{DATASET_NAME}/img/" + "{filepath}"
# DATASET_FEAT_PATH = f"./dataset/{DATASET_NAME}/feat/" + "{filepath}"
