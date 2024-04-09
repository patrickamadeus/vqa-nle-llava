from transformers import LlavaForConditionalGeneration, VipLlavaForConditionalGeneration, AutoModel, AutoProcessor, AutoModelForCausalLM, AutoModelForPreTraining
import torch

runner = [
#     LlavaForConditionalGeneration,
#     LlavaForConditionalGeneration,
#     VipLlavaForConditionalGeneration,
    LlavaForConditionalGeneration
]

models = [
#     "llava-hf/llava-1.5-7b-hf",
#     "llava-hf/llava-1.5-13b-hf",
#     "llava-hf/vip-llava-13b-hf",
    "llava-hf/llava-v1.6-mistral-7b-hf",
]

for i in range(len(runner)):
    _ = runner[i].from_pretrained(
        models[i], 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True, 
    ).to(0)
    processor = AutoProcessor.from_pretrained(models[i])