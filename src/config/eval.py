from src.base import load_config, unpack_json, get_all_filepaths
from src.inference import load_model


PROMPT_DICT = {
    "self_factoid": "./prompt/eval/self_factoid.txt",
    "self_reasoning" : "./prompt/eval/self_reasoning.txt",
    "self_reasoning_split": "./prompt/eval/self_reasoning/",
    "self_factoid_bakllava" : "./prompt/eval/self_factoid_bakllava.txt",
    "self_reasoning_bakllava_split": "./prompt/eval/self_reasoning_bakllava/",
    "openai_story": "./prompt/eval/openai_story.txt"
}


# Load from YAML
# Load from dictionary
eval_cfg                 = load_config("./", "eval.yml")

SEED                    = eval_cfg["seed"]
TEST_NAME               = eval_cfg["test_name"]
EVAL_NUM               = eval_cfg["eval_amount"]
OPENAI_KEY              = eval_cfg["run_params"]["openai_key"]

METADATA_JSON = unpack_json(f"./result/{TEST_NAME}/metadata.json")
DATASET_NAME  = METADATA_JSON["dataset_name"]

MODEL_NAME              = eval_cfg["model"]["name"]
MODEL_PATH              = eval_cfg["model"]["path"]
MODEL_FAMILY            = eval_cfg["model"]["family"]
MODEL_USE_4_BIT         = eval_cfg["model"]["params"]["use_4_bit"]
MODEL_DEVICE            = eval_cfg["model"]["params"]["device"]
MODEL_LOW_CPU           = eval_cfg["model"]["params"]["low_cpu"]
MODEL, PROCESSOR = load_model(
    MODEL_PATH,
    MODEL_FAMILY,
    MODEL_LOW_CPU,
    device = MODEL_DEVICE,
    seed = SEED
)

PROMPT_EVAL_FACTOID_KEY = eval_cfg["prompt"]["factoid"]
PROMPT_EVAL_REASONING_KEY = eval_cfg["prompt"]["reasoning"]
PROMPT_EVAL_FACTOID_PATH = PROMPT_DICT[PROMPT_EVAL_FACTOID_KEY]
PROMPT_EVAL_REASONING_PATH = PROMPT_DICT[PROMPT_EVAL_REASONING_KEY]

DATASET_IMG_PATH = f"./dataset/{DATASET_NAME}/img/" + "{filepath}"
DATASET_FEAT_PATH = f"./dataset/{DATASET_NAME}/feat/" + "{filepath}"


# Gather Output
INTER_JSON = {}
RESULT_JSON = unpack_json(f"./result/{TEST_NAME}/res.json")[:EVAL_NUM]
if PROMPT_EVAL_REASONING_KEY == "openai_story":
    INTER_JSON = unpack_json(f"./result/{TEST_NAME}/misc/inter.json")

# Gather Prompts
PROMPT_EVAL_REASONING_LIST = []
if '.txt' not in PROMPT_EVAL_REASONING_PATH:
    PROMPT_EVAL_REASONING_PATHS, _ = get_all_filepaths(PROMPT_EVAL_REASONING_PATH)

    for path in PROMPT_EVAL_REASONING_PATHS:
        with open(path, 'r') as f:
            PROMPT_EVAL_REASONING_LIST.append(f.read())
else:
    with open(PROMPT_EVAL_REASONING_PATH, 'r') as f:
        PROMPT_EVAL_REASONING_LIST.append(f.read())

PROMPT_EVAL_FACTOID = ""

with open(PROMPT_EVAL_FACTOID_PATH, 'r') as f: 
    PROMPT_EVAL_FACTOID = f.read()    
        




