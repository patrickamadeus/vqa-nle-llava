import gc
import urllib3

import torch
from transformers import (
    LlavaForConditionalGeneration,
    VipLlavaForConditionalGeneration,
    LlavaNextProcessor,
)

runner = [
    LlavaForConditionalGeneration,
    LlavaForConditionalGeneration,
    VipLlavaForConditionalGeneration,
]

models = [
    "llava-hf/llava-1.5-7b-hf",
    "llava-hf/llava-1.5-13b-hf",
    "llava-hf/vip-llava-13b-hf",
]

for i in range(len(runner)):
    retry_limit = 10

    while retry_limit:
        try:
            model = (
                runner[i]
                .from_pretrained(
                    models[i],
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                )
                .to(0)
            )
            processor = LlavaNextProcessor.from_pretrained(models[i])
            break
        except urllib3.exceptions.ProtocolError:
            retry_limit -= 1
            print("ProtocolError occurred. Retrying...")

    del model
    del processor
    gc.collect()
    with torch.no_grad():
        torch.cuda.empty_cache()
