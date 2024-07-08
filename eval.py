import logging
import warnings

from src.base import init_logging
from src.config.eval import (
    TEST_GROUP,
    TEST_NAME,
)
from src.helper.eval_helper import (
    export_eval,
    gen_question_prefix,
    gen_size_hist,
    gwet_AC2,
    gen_subjective_xlsx,
    gen_quant_subj_df,
    gen_subj_rank,
    merge_histogram,
    merge_prefix,
    gen_quant_subj_chart,
    gen_rate_chart,
    gen_dist_analysis,
    gen_prefix_analysis,
)
from tqdm import tqdm

warnings.filterwarnings("ignore")


init_logging()

singleton_pipeline = [
    ("gwet", gwet_AC2, "json"),
#     ("gen_size", gen_size_hist, "plt"),
    ("prefix", gen_question_prefix, "plt"),
    ("gen_xlsx", gen_subjective_xlsx, "xlsx"),
]

multi_eval_pipeline = [
#     ("merge_histogram", merge_histogram, "img"),
    ("merge_distribution", gen_dist_analysis, "img/csv"),
#     ("merge_prefix", merge_prefix, "img"),
    ("merge_prefix", gen_prefix_analysis, "img"),
    ("quantitative_subj_analysis", gen_quant_subj_df, "df"),
    ("rank_analysis", gen_subj_rank, "df"),
    ("merge_quantitative_chart_replace", gen_quant_subj_chart, "plt"),
    ("merge_quantitative_chart_remove", gen_quant_subj_chart, "plt"),
    ("merge_rate_chart", gen_rate_chart, "plt")
]

logging.info("Starting Singleton Evaluation...")
for test_name in TEST_GROUP:
    logging.info(f"[{test_name}] - Inference started...")
    for name, func, mode in tqdm(singleton_pipeline):
        try:
            res = func(test_name)
            export_eval(name, res, test_name, mode)
            logging.info(f"[{test_name}] - [{name}] Inference Successful...")
        except Exception as e:
            logging.error(f"Error occurred while processing '{name}': {str(e)}")
            continue


logging.info("Starting Multi Evaluation...")
logging.info(f"[] - Inference started...")
for name, func, mode in tqdm(multi_eval_pipeline):
    try:
        # For PoC 
        if "remove" in name:
            res = func(test_names = TEST_GROUP, mode = "remove")
        else:
            res = func(test_names = TEST_GROUP)

        # export for non-img files    
        if not name.startswith("merge"):
            export_eval(name, res, mode=mode)
        logging.info(f"[{name}] Inference Successful...")
    except Exception as e:
        logging.error(f"Error occurred while processing '{name}': {str(e)}")
        continue