# Importing built-in package
import torch
from PIL import Image
from math import ceil
from tqdm import tqdm
from datetime import datetime
from time import time
import random
import numpy as np
import torch
import requests
from transformers import LlavaForConditionalGeneration, VipLlavaForConditionalGeneration, AutoModel, AutoProcessor, AutoModelForCausalLM, AutoModelForPreTraining
from .base import set_seed, encode_image


def load_model(model_path, model_family, low_cpu_mem_usage, device = "cuda", seed = 42):
    
    MODEL_LOADER_DICT = {
        "llava" : LlavaForConditionalGeneration,
        "vip_llava": VipLlavaForConditionalGeneration,
        "auto": AutoModel,
        "llama": AutoModelForCausalLM,
        "llava-1.6": AutoModelForPreTraining
    }
    
    model, processor = None, None
    if "openai" not in model_family:
        set_seed(seed)
        model = MODEL_LOADER_DICT[model_family].from_pretrained(
            model_path,
            torch_dtype = torch.float16,
            low_cpu_mem_usage = low_cpu_mem_usage
        ).to(device)
        processor = AutoProcessor.from_pretrained(model_path)
        
    print(f"Loaded {model_path}")
    
    return model, processor


def inference_OpenAI(model, processor, prompt, img_path, api_key=None, data_type = "jpeg", max_tokens = 300):
    base64_image = encode_image(img_path)
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": f"{model}",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{prompt}"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{data_type};base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": f"{max_tokens}"
    }
    
    start_time = time()
    response = requests.post(
        "https://api.openai.com/v1/chat/completions", 
        headers=headers, json=payload
    )
    end_time = time()
    res = response.json()['choices'][0]['message']['content']
    t = end_time - start_time
    
    return res, t


# Inference helpers
def inference_hf(
    model, processor,
    prompt, api_key=None, boilerplate_prompt=True,
    img_path=None, img_raw=None, 
    max_new_tokens=1500, do_sample=False, skip_special_tokens=True) -> (str, float):
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
    
    inputs = processor(prompt, img_raw, return_tensors='pt').to(0, torch.float16)
    raw_output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=do_sample)
    output = processor.decode(raw_output[0], skip_special_tokens=skip_special_tokens)
    
    if boilerplate_prompt:
        output = output[output.index("ASSISTANT:") + 11:]
    
    end_time = time()
    seconds = end_time - start_time
    
    return output+"\n", seconds


def batch_inference_hf(
    model, processor, 
    prompt, img_path, total_pair_count, 
    boilerplate_prompt=True, pair_per_batch=10, 
    max_new_tokens=1500, do_sample=False, skip_special_tokens=True, initial_seed=42):
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
            model, processor, 
            prompt.format(number=pairs_in_batch),
            boilerplate_prompt=boilerplate_prompt,
            img_raw=img_raw,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            skip_special_tokens=skip_special_tokens
        )
        
        total_output += output + "\n"
        total_seconds += seconds
        
    set_seed(initial_seed)
    
    return total_output, total_seconds
