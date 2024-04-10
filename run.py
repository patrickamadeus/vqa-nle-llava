# Methods
import logging

from src.base import (
    get_all_valid_filepaths,
    get_filename,
    init_logging,
    raw_output_splitter,
)
from src.config.run import (
    DATASET_DATA_COUNT,
    DATASET_IMG_PATH,
    DATASET_NAME,
    MODEL,
    MODEL_FAMILY,
    MODEL_NAME,
    PARAM_USE_NONVIS,
    PROCESSOR,
    PROMPT_INTER,
    PROMPT_INTER_KEY,
    PROMPT_IS_MULTISTEP,
    PROMPT_PRIMARY,
    PROMPT_PRIMARY_KEY,
    SCENE_GRAPH,
    TEST_NAME,
)
from src.config.run import (
    PARAM_NUM_PER_INFERENCE as NUM,
)
from src.inference import base_inference_runner, nonvis_inference_runner
from src.parser import export_result, parse_output
from tqdm import tqdm

init_logging()

total_data, total_sec, prev_i = [], 0, 0
primary_raw_out, inter_raw_out, total_inter_data = "", "", {}

img_paths, real_data_count = get_all_valid_filepaths(
    folder_path=DATASET_IMG_PATH, scene_graph=SCENE_GRAPH, n=DATASET_DATA_COUNT
)

logging.info("Starting Generation...")

annot_metadatas = []
for img_path in tqdm(img_paths):
    img_id_ext, img_id = get_filename(img_path)
    runner_config = {
        "pair_num": NUM,
        "is_multistep": PROMPT_IS_MULTISTEP,
    }

    if PARAM_USE_NONVIS:
        inference_runner = nonvis_inference_runner
    else:
        inference_runner = base_inference_runner

    primary_out, primary_sec, inter_out, inter_sec, metadata = inference_runner(
        model=MODEL,
        processor=PROCESSOR,
        prompt_primary=PROMPT_PRIMARY,
        prompt_inter=PROMPT_INTER,
        img_path=img_path,
        scene_graph=SCENE_GRAPH,
        runner_config=runner_config,
    )

    parsed_data = parse_output(primary_out, img_id_ext, prev_i)

    primary_raw_out += raw_output_splitter(img_id_ext, primary_out)
    inter_raw_out += raw_output_splitter(img_id_ext, inter_out)
    total_sec += primary_sec + inter_sec
    prev_i += len(parsed_data)
    total_data += parsed_data
    if PARAM_USE_NONVIS:
        annot_metadatas.append(metadata)

    logging.info(
        f"[{img_id_ext}] - success generated {len(parsed_data)} synthetic data(s)"
    )

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
    annot_metadatas=annot_metadatas,
)
logging.info("Program exited.")
