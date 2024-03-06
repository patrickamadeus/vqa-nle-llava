from ..base import load_config
from ..inference import load_model
import json
import os

PROMPT_DICT = {
    "qg_lr": "./prompt/qg_lr/",
    "lr": "./prompt/lr/",
    "story": "./prompt/qg_story/story_base.txt",
    "story_long": "./prompt/qg_story/story_long.txt",
    "qg_story": "./prompt/qg_story/gen_base.txt",
    "qg_story_limit": "./prompt/qg_story/gen_limit.txt",
    "naive": "./prompt/naive/base.txt",
    "clevr_base": "./prompt/CLEVR/base.txt",
    "vlm_korika": "./prompt/vlm_korika.txt"
}



try: 
    # Load from YAML
    # Load from dictionary
    run_cfg                 = load_config("./", "run.yml")
    
    SEED                    = run_cfg["seed"]

    DATASET_NAME            = run_cfg["dataset"]["name"]
    DATASET_DATA_COUNT      = int(run_cfg["dataset"]["count"])

    MODEL_NAME              = run_cfg["model"]["name"]
    MODEL_PATH              = run_cfg["model"]["path"]
    MODEL_FAMILY            = run_cfg["model"]["family"]
    MODEL_USE_4_BIT         = run_cfg["model"]["params"]["use_4_bit"]
    MODEL_DEVICE            = run_cfg["model"]["params"]["device"]
    MODEL_LOW_CPU           = run_cfg["model"]["params"]["low_cpu"]
    MODEL, PROCESSOR = load_model(
        MODEL_PATH,
        MODEL_FAMILY,
        MODEL_LOW_CPU,
        device = MODEL_DEVICE,
        seed = SEED
    )
    
    PROMPT_IS_MULTISTEP     = int(run_cfg["prompt"]["multistep"])
    PROMPT_PRIMARY_KEY      = run_cfg["prompt"]["primary"]
    PROMPT_INTER_KEY    = run_cfg["prompt"]["inter"]
    PROMPT_PRIMARY_PATH     = PROMPT_DICT[PROMPT_PRIMARY_KEY]
    PROMPT_INTER_PATH   = ""
    if PROMPT_IS_MULTISTEP:
        PROMPT_INTER_PATH = PROMPT_DICT[PROMPT_INTER_KEY]

    TEST_NAME               = run_cfg["test_name"]
    PARAM_NUM_PER_INFERENCE = int(run_cfg["run_params"]["num_per_inference"])
    PARAM_USE_NONVIS        = int(run_cfg["run_params"]["use_nonvis"])
    PARAM_USE_EXTS= run_cfg["run_params"]["use_img_ext"]
    
    
    # Derived config constants
    # DATASET
    DATASET_IMG_PATH        = f"./dataset/{DATASET_NAME}/img"
    DATASET_FEAT_PATH       = f"./dataset/{DATASET_NAME}/feat"

    # PROMPT
    PROMPT_PRIMARY          = ""
    PROMPT_INTER       = ""
    
    with open(PROMPT_PRIMARY_PATH) as f: PROMPT_PRIMARY = f.read()
    if PROMPT_IS_MULTISTEP:
        with open(PROMPT_INTER_PATH) as f: PROMPT_INTER = f.read()
    
    
except Exception as e:
    print(f"Config loading error!\n{e}")




