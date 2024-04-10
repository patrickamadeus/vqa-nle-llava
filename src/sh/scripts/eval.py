from transformers import AutoProcessor, LlavaForConditionalGeneration

models = ["llava-hf/llava-1.5-13b-hf"]

for model in models:
    _ = LlavaForConditionalGeneration.from_pretrained(
        model,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(0)
    processor = AutoProcessor.from_pretrained(model)
