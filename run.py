# Methods
from src.parser import parse_output, export_result
from src.inference import inference_hf
from src.base import (
    get_all_filepaths,
    get_filename,
    init_logging,
)
from tqdm import tqdm
import logging

from src.config.run import (
    # test folder result name
    TEST_NAME,
    # Dataset
    DATASET_IMG_PATH,
    DATASET_DATA_COUNT,
    DATASET_NAME,
    # Model
    MODEL_NAME,
    MODEL_FAMILY,
    MODEL,
    PROCESSOR,
    # Prompts
    PROMPT_PRIMARY,
    PROMPT_INTER,
    PROMPT_IS_MULTISTEP,
    PROMPT_PRIMARY_KEY,
    PROMPT_INTER_KEY,
    # Params
    PARAM_NUM_PER_INFERENCE as NUM,
)

init_logging()

total_data, total_sec, prev_i = [], 0, 0
primary_raw_out = ""

total_inter_data = {}
inter_raw_out = ""

img_paths, real_data_count = get_all_filepaths(DATASET_IMG_PATH, DATASET_DATA_COUNT)

logging.info("Starting Generation...")
for img_path in tqdm(img_paths):
    # prepare image
    img_id = get_filename(img_path, extension=True)
    logging.info(f"[{img_id}] - Inference started...")

    # Multistep / single-step inference
    inter_output = ""
    inter_sec = 0
    if PROMPT_IS_MULTISTEP:
        inter_output, inter_sec = inference_hf(
            MODEL, PROCESSOR, PROMPT_INTER.format(number=NUM), img_path=img_path
        )
        inter_raw_out += f"{img_id}\n--------\n{inter_output}\n"
        total_inter_data[img_id] = inter_output
        logging.info(f"[{img_id}] - Inter Inference finished ({inter_sec}s)")

    primary_output, primary_sec = inference_hf(
        MODEL,
        PROCESSOR,
        PROMPT_PRIMARY.format(number=NUM, intermediary=inter_output),
        img_path=img_path,
    )
    logging.info(f"[{img_id}] - Primary Inference finished ({primary_sec}s)")

    # Parsing & stat update
    primary_raw_out += f"{img_id}\n--------\n{primary_output}\n"
    parsed_data = parse_output(primary_output, img_id, prev_i)

    total_data += parsed_data
    prev_i += len(parsed_data)
    total_sec += primary_sec + inter_sec

    logging.info(f"[{img_id}] - success generated {len(parsed_data)} synthetic data(s)")


export_result(
    data=total_data,
    total_data=real_data_count * NUM,
    total_gen=prev_i,
    total_sec=total_sec,
    primary_raw_out=primary_raw_out,
    inter_raw_out=inter_raw_out,
    test_name=TEST_NAME,
    dataset_name=DATASET_NAME,
    model_name=MODEL_NAME,
    model_family=MODEL_FAMILY,
    prompt_primary=PROMPT_PRIMARY_KEY,
    prompt_inter=PROMPT_INTER_KEY,
    inter_dict=total_inter_data,
)
logging.info("Program exited.")
