# Methods
import logging
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
from PIL import Image
import os
import json
from sentence_transformers import SentenceTransformer, util
import re

from src.base import (
    get_filename,
    init_logging,
    raw_output_splitter,
    unpack_json,
    validate_question,
    validate_short_answer,
    validate_reason,
    get_all_valid_filepaths,
    expand_prefix_stratify,
)

from src.inference import(
    inference_hf,
    load_model,
)

from src.parser import parse_output
init_logging()


########### ########### ########### ########### ########### 
########### ########### ########### ########### ###########
sc_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

TEST_NAME = "10-4_vicuna13_naive-optim_500"
MODEL,PROCESSOR = load_model(
    model_path = "llava-hf/llava-1.5-13b-hf",
    model_family = "llava",
    low_cpu_mem_usage = True
)

DATASET_DATA_COUNT = 200
PARAM_NUM_PER_INFERENCE = 1
PREFIXES = ["what", "is/are (pick one that fits the most)", "which", "how many", "where"]
PREFIXES_PROPORTIONS = [3,2,1,1,1]
PREFIX_LIST = expand_prefix_stratify(PREFIXES, PREFIXES_PROPORTIONS, DATASET_DATA_COUNT * PARAM_NUM_PER_INFERENCE)


SCENE_GRAPH_PATH = "./dataset/feat/sceneGraphs.json"
with open(SCENE_GRAPH_PATH) as json_file:
    SCENE_GRAPH = json.load(json_file)

    
Q_PROMPT = "./prompt/self_consistency/question.txt"
A_PROMPT = "./prompt/self_consistency/short_answer.txt"
R_PROMPT = "./prompt/self_consistency/reasoning.txt"
R_COT_PROMPT = "./prompt/self_consistency/reasoning_cot.txt"
R_REACT_PROMPT = "./prompt/self_consistency/react.txt"


with open(Q_PROMPT,"r") as f1, \
     open(A_PROMPT,"r") as f2, \
     open(R_PROMPT,"r") as f3, \
     open(R_COT_PROMPT,"r") as f4, \
     open(R_REACT_PROMPT,"r") as f5:
    Q_PROMPT = f1.read()
    A_PROMPT = f2.read()
    R_PROMPT = f3.read()
    R_COT_PROMPT = f4.read()
    R_REACT_PROMPT = f5.read()


def self_consistency_scrape_question(test_name):
    json_path = f"./result/{test_name}/res.json"
    json = unpack_json(json_path)
    
    img_q_pairs = [
        (
            f"./dataset/img/{data['img_id']}", 
            data["question"]
        ) for data in json
    ]
    
    return img_q_pairs


def get_top_answer(answers):
    if len(answers) == 1:
        return answers[0]

    embeddings = [
        sc_model.encode(sentence, convert_to_tensor=True) for sentence in answers
    ]
    
    scores = []
    max_score = -1
    max_i = 0
    
    for i in range(len(embeddings)):
        score = 0

        for j in range(len(embeddings)):
            if i != j:
                score += util.pytorch_cos_sim(embeddings[i], embeddings[j])
        
        scores.append(score)
        if score / (len(embeddings) - 1) > max_score:
            max_score = score
            max_i = i
    
    return answers[max_i]


def generate_question(model, processor, img, prompt, prefix):
    q, sec = inference_hf(
        model, processor, 
        prompt.format(prefix = prefix), 
        img_raw = img,
        max_new_tokens = 20,
    )
    
    return validate_question(q), sec


def generate_short_answer(model, processor, img, prompt, question):
    a, sec = inference_hf(
        model, processor, 
        prompt.format(question = question), 
        img_raw = img,
        max_new_tokens = 25,
    )
    
    return validate_short_answer(a), sec

def extract_reason_react(text):
    observation_pattern = r'Observation:(.*?)Thoughts:'
    thoughts_pattern = r'Thoughts:(.*?)Action:'
    action_pattern = r'Action:(.*?)Reason:'

    observation_match = re.search(observation_pattern, text, re.DOTALL)
    thoughts_match = re.search(thoughts_pattern, text, re.DOTALL)
    action_match = re.search(action_pattern, text, re.DOTALL)

    if observation_match and thoughts_match and action_match:
        reason_pattern = r'Reason:(.*?)$'
        reason_match = re.search(reason_pattern, action_match.group(1).strip(), re.DOTALL)
        if reason_match:
            return reason_match.group(1).strip()

    return text.strip()
    


def generate_reasons(model, processor, img, prompts, question, short_answer, max_new_tokens = [70,70,300]):
    reasons = []
    raw_reasons = []
    total_sec = 0
    
    for i in range(len(prompts)):
        r, sec = inference_hf(
            model, processor, 
            prompts[i].format(question = question, short_answer = short_answer), 
            img_raw = img,
            max_new_tokens = max_new_tokens[i],
        )
        raw_reasons.append(r)
        
        if i == len(prompts) - 1:
            r = extract_reason_react(r)

        reasons.append(validate_reason(r))
        total_sec += sec
    
    return reasons, raw_reasons, total_sec

########### ########### ########### ########### ###########
########### ########### ########### ########### ###########


def pipeline(scrape_config = None):
    if scrape_config is not None:
        img_question_pairs = self_consistency_scrape_question(scrape_config["test_name"])
    else:
        img_paths, real_data_count = get_all_valid_filepaths(
            folder_path="./dataset/img", scene_graph=SCENE_GRAPH, n=DATASET_DATA_COUNT
        )
        img_question_pairs = [(img_path,None) for img_path in img_paths]
    
    
    total_data, prev_i, total_raw_out = [], 0, ""
    R_PROMPTS = [R_PROMPT, R_COT_PROMPT, R_REACT_PROMPT]
    i_prefix = 0
    
    total_time = 0
    
    for img_path, question in tqdm(img_question_pairs[:]):
        img_id_ext, _ = get_filename(img_path)
        img = Image.open(img_path)
        
        question = question
        
        sec_0 = 0
        if scrape_config is None:
            question, sec_0 = generate_question(
                MODEL, PROCESSOR, 
                img,
                Q_PROMPT,
                prefix = PREFIX_LIST[i_prefix]
            )
        
        short_answer, sec_1 = generate_short_answer(
            MODEL, PROCESSOR, 
            img, 
            A_PROMPT, 
            question
        )
        reasons, raw_reasons, sec_2 = generate_reasons(
            MODEL, PROCESSOR, 
            img, 
            R_PROMPTS,
            question,
            short_answer
        )
        
        top_reason = get_top_answer(reasons)
        output = f"Question: {question}\nShort Answer: {short_answer}\nReason: {top_reason}\n"
        parsed_data = parse_output(output, img_id_ext, prev_i)
        
        total_time = total_time + sec_0 + sec_1 + sec_2
        prev_i += len(parsed_data)
        i_prefix += 1
        
        total_data += parsed_data
        total_raw_out += raw_output_splitter(img_id_ext, output, extras = raw_reasons)
        logging.info(
            f"[{img_id_ext}] - success generated {len(parsed_data)} synthetic data(s)"
        )
        
    
    total_raw_out += f"\n\nTime taken: {total_time}s\nTotal Generated: {prev_i}/200"
    raw_path = os.path.join("./result/20-5_vicuna13_sc_200/raw.txt")
    res_path = os.path.join("./result/20-5_vicuna13_sc_200/res.json")
    with open(res_path, "w") as res_file, open(raw_path, "w") as raw_file:
        json.dump(total_data, res_file, indent=2)
        raw_file.write(total_raw_out)
    



if __name__ == "__main__":
    logging.info("Starting Generation...")
    
#     scrape_config = {"test_name": TEST_NAME}
    scrape_config = None
    
    pipeline(scrape_config = scrape_config)
        
        