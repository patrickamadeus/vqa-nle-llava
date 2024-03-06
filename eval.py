from src.parser import export_eval
from src.inference import inference_hf, inference_OpenAI
from src.base import init_logging
from tqdm import tqdm
import logging
import os
import time

# Config Variable
from src.config.eval import (
    # test folder result name
    TEST_NAME, MODEL_PATH, MODEL_FAMILY,
    # Data for testing
    DATASET_IMG_PATH, RESULT_JSON,
    # Prompts
    PROMPT_EVAL_FACTOID, PROMPT_EVAL_REASONING_LIST, 
    PROMPT_EVAL_FACTOID_KEY, PROMPT_EVAL_REASONING_KEY,
    # Model and Processor
    MODEL, PROCESSOR, OPENAI_KEY
)

init_logging()

ids, factoid_scores, reasoning_scores = [], [], []
subject_ids, img_ids, times = [],[],[]

total_raw_output = "<PROMPTS>\n" + PROMPT_EVAL_FACTOID + "\n" + "\n\n".join(PROMPT_EVAL_REASONING_LIST)

inference_func = inference_hf
if MODEL_FAMILY == "openai":
    inference_func = inference_OpenAI
    MODEL = MODEL_PATH  
    
total_time = 0

for tc in tqdm(RESULT_JSON):
    tc_id = tc["id"]
    tc_img_id = tc["img_id"]
    q = tc["question"]
    sa = tc["short_answer"]
    ra = tc["reasoned_answer"]
    
    img_path = DATASET_IMG_PATH.format(filepath = tc_img_id)
    if not os.path.isfile(img_path):
        print(f"{img_path} is an invalid file.")
        continue
    
    raw_factoid_eval_res, factoid_sec = inference_func(
        MODEL, PROCESSOR,
        PROMPT_EVAL_FACTOID.format(question = q, answer = sa),
        img_path = img_path,
        api_key = OPENAI_KEY
    )
    logging.info(f"[{tc_img_id}] - Factoid Eval finished ({factoid_sec}s)")
    
    temp_reasoning_scores = []
    total_r_sec = 0
    for reasoning_prompt in PROMPT_EVAL_REASONING_LIST:
#         time.sleep(2)
        reasoning_eval_res, r_sec = inference_func(
            MODEL, PROCESSOR,
            reasoning_prompt.format(question = q, answer = sa, reason = ra),
            img_path = img_path,
            api_key = OPENAI_KEY
        )
        temp_reasoning_scores.append(reasoning_eval_res)
        total_r_sec += r_sec
    
    joined_temp_reasoning_scores = ';'.join(temp_reasoning_scores)
    
    ids.append(tc_id)
    factoid_scores.append(raw_factoid_eval_res)
    logging.info(f"[{tc_img_id}] - Reasoning Eval finished ({total_r_sec}s)")
    reasoning_scores.append(joined_temp_reasoning_scores)
    
    total_time += total_r_sec + factoid_sec
    
    total_raw_output += f"""
id
{tc_id}
FACTOID
{raw_factoid_eval_res}
R0
{temp_reasoning_scores[0]}
R1
{temp_reasoning_scores[1]}
R2
{temp_reasoning_scores[2]}
R3
{temp_reasoning_scores[3]}
R4
{temp_reasoning_scores[4]}
\n\n
"""
    
    
export_eval(
    TEST_NAME,
    MODEL_PATH,
    ids, 
    factoid_scores,
    reasoning_scores,
    total_raw_output,
    total_time,
    PROMPT_EVAL_FACTOID_KEY, 
    PROMPT_EVAL_REASONING_KEY
)
    