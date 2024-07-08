import json

from src.base import load_config, expand_prefix_stratify
from src.legacy.inference import load_model

PROMPT_DICT = {
    "story": "./prompt/qg_story/story_base.txt",
    "story_long": "./prompt/qg_story/story_long.txt",
    "qg_story": "./prompt/qg_story/gen_base.txt",
    "qg_story_limit": "./prompt/qg_story/gen_limit.txt",
    "qg_story_optim": "./prompt/qg_story/gen_optim.txt",
    "naive": "./prompt/naive/base.txt",
    "naive_optim": "./prompt/naive/optim.txt",
    "naive_random": "./prompt/naive/random.txt",
    "nonvis_base": "./prompt/nonvis/base.txt",
    "nonvis_optim": "./prompt/nonvis/optim.txt",
    "sc_question": "./prompt/self_consistency/question.txt",
    "sc_short_answer": "./prompt/self_consistency/short_answer.txt",
    "sc_reason": "./prompt/self_consistency/reasoning.txt"
}


# Load from YAML
# Load from dictionary
run_cfg = load_config("./", "run.yml")

SEED = run_cfg["seed"]

DATASET_NAME = run_cfg["dataset"]["name"]
DATASET_DATA_COUNT = int(run_cfg["dataset"]["count"])
SCENE_GRAPH_PATH = "./dataset/feat/sceneGraphs.json"

MODEL_NAME = run_cfg["model"]["name"]
MODEL_PATH = run_cfg["model"]["path"]
MODEL_FAMILY = run_cfg["model"]["family"]
MODEL_USE_4_BIT = run_cfg["model"]["params"]["use_8_bit"]
MODEL_DEVICE = run_cfg["model"]["params"]["device"]
MODEL_LOW_CPU = run_cfg["model"]["params"]["low_cpu"]
MODEL, PROCESSOR = load_model(
    MODEL_PATH, MODEL_FAMILY, MODEL_LOW_CPU, device=MODEL_DEVICE, seed=SEED
)

PROMPT_IS_MULTISTEP = int(run_cfg["prompt"]["multistep"])
PROMPT_PRIMARY_KEY = run_cfg["prompt"]["primary"]
PROMPT_INTER_KEY = run_cfg["prompt"]["inter"]
PROMPT_PRIMARY_PATH = PROMPT_DICT[PROMPT_PRIMARY_KEY]
PROMPT_INTER_PATH = ""
if PROMPT_IS_MULTISTEP:
    PROMPT_INTER_PATH = PROMPT_DICT[PROMPT_INTER_KEY]

TEST_NAME = run_cfg["test_name"]
PARAM_NUM_PER_INFERENCE = int(run_cfg["run_params"]["num_per_inference"])
PARAM_USE_NONVIS = int(run_cfg["run_params"]["use_nonvis"])
with open(SCENE_GRAPH_PATH) as json_file:
    SCENE_GRAPH = json.load(json_file)
PARAM_USE_EXTS = run_cfg["run_params"]["use_img_ext"]


# Derived config constants
# DATASET
DATASET_IMG_PATH = "./dataset/img"
DATASET_FEAT_PATH = "./dataset/feat"

# PROMPT
PROMPT_PRIMARY = ""
PROMPT_INTER = ""

with open(PROMPT_PRIMARY_PATH) as f:
    PROMPT_PRIMARY = f.read()
if PROMPT_IS_MULTISTEP:
    with open(PROMPT_INTER_PATH) as f:
        PROMPT_INTER = f.read()

# QUESTION PREFIXES
PREFIXES = ["what", "is/am/are (pick one that fits the most)", "which", "how many", "where/when (pick one that fits the most)", "who", "whose/whom (pick one that fits the most)"]
PREFIXES_PROPORTIONS = [2,2,2,2,1,1,1]

if PARAM_USE_NONVIS:
    PREFIXES = PREFIXES[:5]
    PREFIXES_PROPORTIONS = PREFIXES_PROPORTIONS[:5]

PREFIX_LIST = expand_prefix_stratify(PREFIXES, PREFIXES_PROPORTIONS, DATASET_DATA_COUNT * PARAM_NUM_PER_INFERENCE)
