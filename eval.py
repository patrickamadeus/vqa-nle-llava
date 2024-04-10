from src.eval_helper import (
    gwet_AC2,
    gen_size_hist,
    gen_question_prefix,
    export_eval,
    TEST_NAME,
)
from src.base import init_logging

from tqdm import tqdm
import logging
import warnings

warnings.filterwarnings("ignore")


init_logging()

pipeline = [
    ("gwet", gwet_AC2, "json"),
    ("gen_size", gen_size_hist, "plt"),
    ("prefix", gen_question_prefix, "plt"),
]

logging.info("Starting Evaluation...")
logging.info(f"[{TEST_NAME}] - Inference started...")
for name, func, mode in tqdm(pipeline):
    try:
        res = func(TEST_NAME)
        export_eval(name, res, mode)
        logging.info(f"[{TEST_NAME}] - [{name}] Inference Successful...")
    except Exception as e:
        logging.error(f"Error occurred while processing '{name}': {str(e)}")
        continue
