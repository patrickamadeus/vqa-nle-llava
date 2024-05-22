# Importing built-in package
import logging
from math import ceil
from time import time

import requests
import torch
from PIL import Image
from tqdm import tqdm
import json
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForPreTraining,
    AutoProcessor,
    LlavaForConditionalGeneration,
    VipLlavaForConditionalGeneration,
    LlavaNextProcessor, 
    LlavaNextForConditionalGeneration
)

from src.base import (
    annotate_images,
    encode_image,
    get_filename,
    set_seed,
    unpack_json,
)


def load_model(model_path, model_family, low_cpu_mem_usage, device="cuda", seed=42, load_in_8bit = False):
    MODEL_LOADER_DICT = {
        "llava": LlavaForConditionalGeneration,
        "vip_llava": VipLlavaForConditionalGeneration,
        "auto": AutoModel,
        "llama": AutoModelForCausalLM,
        "llava-1.6": LlavaNextForConditionalGeneration,
    }

    model, processor = None, None
    if "openai" not in model_family:
        set_seed(seed)
        
        if load_in_8bit:
            model = (
                MODEL_LOADER_DICT[model_family]
                .from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=low_cpu_mem_usage,
                    load_in_8bit=load_in_8bit,
#                     use_flash_attention_2=True
                    attn_implementation="flash_attention_2"
                )
            )
        else:
            model = (
                MODEL_LOADER_DICT[model_family]
                .from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=low_cpu_mem_usage,
#                     use_flash_attention_2=True
                    attn_implementation="flash_attention_2"
                )
                .to(device)
            )  
        processor = AutoProcessor.from_pretrained(model_path)

    print(f"Loaded {model_path}")

    return model, processor


def inference_OpenAI(
    model, processor, prompt, img_path, api_key=None, data_type="jpeg", max_tokens=300
):
    base64_image = encode_image(img_path)

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    payload = {
        "model": f"{model}",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{prompt}"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{data_type};base64,{base64_image}"
                        },
                    },
                ],
            }
        ],
        "max_tokens": f"{max_tokens}",
    }

    start_time = time()
    response = requests.post(
        "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
    )
    end_time = time()
    res = response.json()["choices"][0]["message"]["content"]
    t = end_time - start_time

    return res, t


def inference_hf(
    model,
    processor,
    prompt,
    api_key=None,
    boilerplate_prompt=True,
    img_path=None,
    img_raw=None,
    max_new_tokens=1500,
    do_sample=False,
    skip_special_tokens=True,
) -> (str, float):
    """
    Perform inference using a HuggingFace model.

    Args:
    - model: The HuggingFace model.
    - processor: The HuggingFace processor.
    - prompt (str): The input prompt.
    - boilerplate_prompt (bool): Whether to include boilerplate prompt.
    - img_path (str): Path to the image.
    - img_raw: Raw image data.
    - max_new_tokens (int): Maximum number of new tokens.
    - do_sample (bool): Whether to sample.
    - skip_special_tokens (bool): Whether to skip special tokens.

    Returns:
    - Tuple[str, int]: The generated output and the inference time in seconds.
    """
    start_time = time()
    if img_raw is None:
        try:
            img_raw = Image.open(img_path)
        except Exception as e:
            return str(e)

    if boilerplate_prompt:
        prompt = "USER: <image>\n" + prompt + "\nASSISTANT:"

    inputs = processor(
        prompt, img_raw, return_tensors="pt").to(0, torch.float16)
    
    raw_output = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=do_sample
    )
    output = processor.decode(raw_output[0], skip_special_tokens=skip_special_tokens)

    if boilerplate_prompt:
        output = output[output.index("ASSISTANT:") + 11 :]

    end_time = time()
    seconds = end_time - start_time

    return output + "\n", seconds


def inference_hf_multi(
    model,
    processor,
    prompt,
    top_k=50,
    top_p=0.95,
    num_return_sequences=5,
    boilerplate_prompt=True,
    img_path=None,
    img_raw=None,
    max_new_tokens=1500,
    do_sample=True,
    skip_special_tokens=True,
) -> (str, float):
    start_time = time()
    if img_raw is None:
        try:
            img_raw = Image.open(img_path)
        except Exception as e:
            return str(e)

    if boilerplate_prompt:
        prompt = "USER:\n" + prompt + "\nASSISTANT:"

    inputs = processor(prompt, img_raw, return_tensors="pt").to(0, torch.float16)
    
    raw_outputs = model.generate(
        **inputs, 
        max_new_tokens=max_new_tokens, 
        do_sample=do_sample,
        top_k=top_k,
        top_p=top_p,
        num_return_sequences=num_return_sequences
    )
    
    res = []
    for i, sample_output in enumerate(raw_outputs):
        output = processor.decode(sample_output, skip_special_tokens=skip_special_tokens)

        if boilerplate_prompt:
            output = output[output.index("ASSISTANT:") + 11 :]
            
        res.append(output)

    end_time = time()
    seconds = end_time - start_time
    
    return res, seconds


def batch_inference_hf(
    model,
    processor,
    prompt,
    img_path,
    total_pair_count,
    boilerplate_prompt=True,
    pair_per_batch=10,
    max_new_tokens=1500,
    do_sample=False,
    skip_special_tokens=True,
    initial_seed=42,
):
    """
    Perform batched inference using a HuggingFace model.

    Args:
    - model: The HuggingFace model.
    - processor: The HuggingFace processor.
    - prompt (str): The input prompt with a placeholder for the number.
    - img_path (str): Path to the image.
    - total_pair_count (int): Total number of pairs.
    - boilerplate_prompt (bool): Whether to include boilerplate prompt.
    - pair_per_batch (int): Number of pairs per batch.
    - max_new_tokens (int): Maximum number of new tokens.
    - do_sample (bool): Whether to sample.
    - skip_special_tokens (bool): Whether to skip special tokens.
    - initial_seed (int): Initial seed for randomization.

    Returns:
    - Tuple[str, int]: The concatenated output and the total inference time in seconds.
    """
    img_raw = Image.open(img_path)
    num_per_batch = ceil(total_pair_count / pair_per_batch)
    num_last_batch = total_pair_count % pair_per_batch

    total_output = ""
    total_seconds = 0

    for batch in tqdm(range(num_per_batch)):
        if batch == num_per_batch - 1:
            pairs_in_batch = num_last_batch
        else:
            pairs_in_batch = pair_per_batch

        output, seconds = inference_hf(
            model,
            processor,
            prompt.format(number=pairs_in_batch),
            boilerplate_prompt=boilerplate_prompt,
            img_raw=img_raw,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            skip_special_tokens=skip_special_tokens,
        )

        total_output += output + "\n"
        total_seconds += seconds

    set_seed(initial_seed)

    return total_output, total_seconds


def base_inference_runner(
    model,
    processor,
    prompt_primary,
    prompt_inter,
    img_path,
    runner_config: dict,
    scene_graph=None,
):
    img_id_ext, _ = get_filename(img_path)

    logging.info(f"[{img_id_ext}] - Inference started...")

    inter_out, inter_sec = "", 0
    if runner_config["is_multistep"]:
        inter_out, inter_sec = inference_hf(
            model,
            processor,
            prompt_inter.format(number=runner_config["pair_num"]),
            img_path=img_path,
        )
        logging.info(f"[{img_id_ext}] - Inter Inference finished ({inter_sec}s)")

    outs = []
    total_sec = 0
    for prefix in runner_config["prefixes"]:
        primary_out, primary_sec = inference_hf(
            model,
            processor,
            prompt_primary.format(
                intermediary=inter_out,
                prefix=prefix,
            ),
            img_path=img_path,
        )
        outs.append(primary_out)
        total_sec += primary_sec

    logging.info(f"[{img_id_ext}] - Primary Inference finished ({total_sec}s)")

    return "\n".join(outs), total_sec, inter_out, inter_sec, None


def nonvis_inference_runner(
    model,
    processor,
    prompt_primary,
    img_path,
    scene_graph,
    runner_config: dict,
    prompt_inter="",
):
    outs = []
    total_sec = 0
    img_id, complete_annot_tensor = None, None
    
    img_id_ext, img_id = get_filename(img_path)
    
    try:
        raw_objs, complete_annot_tensor = annotate_images(
            img_path, scene_graph[img_id], num_obj=runner_config["pair_num"]
        )

        logging.info(f"[{img_id_ext}] - Nonvis Inference started...")

        for i, obj in enumerate(raw_objs):
            out, sec = inference_hf(
                model,
                processor,
                prompt_primary.format(
                    number=i + 1, name=obj[0], prefix=runner_config["prefixes"][i]
                ),
                img_raw=obj[1],
            )
            outs.append(out)
            total_sec += sec

        logging.info(f"[{img_id_ext}] - Nonvis Inference finished ({sec}s)")
    except Exception as e:
        logging.error(f"[{img_id_ext}] - An unexpected error occurred: {str(e)}")

    return (
        "\n".join(outs),
        total_sec,
        "",
        0,
        {"id": img_id, "complete_tensor": complete_annot_tensor},
    )


def self_consistency_processor(img_path, question, top_k_prompt, conclude_prompt, top_k_sample = 5):
    # Generate top_k_sample answers
    answers, sec1 = inference_hf_multi(
        model, processor, 
        top_k_prompt.format(question = question), 
        img_path = img_path,
        num_return_sequences = top_k_sample
    )
    
    num = top_k_sample
    ques = q
    ans = ""

    for i, answer in enumerate(answers):
        ans += f"{i+1}. {answer}\n\n"
    
    # Conclude majority answer that is the most consistent
    final, sec2 = inference_hf(
        model, processor, 
        conclude.format(number = num, question = ques, answers = ans),
        img_raw = None
    )
    
    return final, sec1+sec2+sec3



def self_consistency_inference_runner(
    model,
    processor,
    top_k_prompt,
    conclude_prompt,
    img_path,
):
    img_id_ext, _ = get_filename(img_path)

    logging.info(f"[{img_id_ext}] - Inference started...")
    
    

    outs = []
    total_sec = 0
    for prefix in runner_config["prefixes"]:
        primary_out, primary_sec = inference_hf(
            model,
            processor,
            prompt_primary.format(
                intermediary=inter_out,
                prefix=prefix,
            ),
            img_path=img_path,
        )
        outs.append(primary_out)
        total_sec += primary_sec

    logging.info(f"[{img_id_ext}] - Primary Inference finished ({total_sec}s)")

    return "\n".join(outs), total_sec, inter_out, inter_sec, None