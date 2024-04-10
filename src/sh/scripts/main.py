from transformers import LlavaForConditionalGeneration, VipLlavaForConditionalGeneration, AutoModel, AutoProcessor, AutoModelForCausalLM, AutoModelForPreTraining
import torch
import gc

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
    # Download model
    model = runner[i].from_pretrained(
        models[i], 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True, 
    ).to(0)
    processor = AutoProcessor.from_pretrained(models[i])
    
    # Free CUDA memory
    del model
    del processor
    gc.collect()
    with torch.no_grad():
        torch.cuda.empty_cache()
