import logging
import warnings

from src.base import init_logging
from src.eval_helper import (
    export_eval,
    gen_quant_subj_df,
    gen_subj_rank
)
from tqdm import tqdm

warnings.filterwarnings("ignore")


RESULT_PATH = "./"
TEST_NAMES = [
    "10-4_vicuna7_naive-optim_500",
    "10-4_vicuna13_naive-optim_500",
    "10-4_vicuna13_qg-story-optim_500",
    "10-4_vicuna13-vip_nonvis-optim_500"
]

map_index = {
    "10-4_vicuna13-vip_nonvis-optim_500" : "13b_nonvis",
    "10-4_vicuna7_naive-optim_500": "7b_naive",
    "10-4_vicuna13_naive-optim_500": "13b_naive",
    "10-4_vicuna13_qg-story-optim_500": "13b_qg-story",
}

pipeline = [
    ("quantitative_subj_analysis", gen_quant_subj_df, "df"),
    ("rank_analysis", gen_subj_rank, "df"),
]


logging.info("Starting Evaluation...")
logging.info(f"[] - Inference started...")
for name, func, mode in tqdm(pipeline):
    # try:
    res = func(test_names = TEST_NAMES, map_index = map_index)
    export_eval(name, res, mode, RESULT_PATH)
    logging.info(f"[{name}] Inference Successful...")
    # except Exception as e:
    #     logging.error(f"Error occurred while processing '{name}': {str(e)}")
    #     continue
