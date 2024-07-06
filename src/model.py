# Importing built-in package
import logging
from time import time

import torch
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    VipLlavaForConditionalGeneration,
    LlavaNextForConditionalGeneration,
)

from src.config.run_class import RunConfig


Config= RunConfig()
ModelConfig = Config.get_model_config()


class LVLM:
    def __init__(self, config=ModelConfig) -> None:
        model_loader = {
            "llava": LlavaForConditionalGeneration,
            "vip_llava": VipLlavaForConditionalGeneration,
            "llava-1.6": LlavaNextForConditionalGeneration,
        }

        self.name = config["name"]
        self.path = config["path"]
        self.__loader = model_loader[config["family"]]
        self.__use_8_bit = config["use_8_bit"]
        self.__device = config["device"]
        self.__low_cpu = config["low_cpu"]

        self.__model = self.__load_model()
        self.__processor = self.__load_processor()

    def __load_model(self):
        self.__model = self.__loader.from_pretrained(
            self.path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=self.__low_cpu,
            load_in_8bit=self.__use_8_bit
        ).to(self.__device)

        logging.info(f"Model {self.name} loaded successfully")

        return self.__model

    def __load_processor(self):
        self.__processor = AutoProcessor.from_pretrained(self.path)

        logging.info(f"Processor for {self.name} loaded successfully")

        return self.__processor

    def __validate_prompt_token(self, prompt):
        if "USER:" in prompt and "ASSISTANT:" in prompt and "<image>" in prompt:
            return prompt
        else:
            prompt = f"USER: <image>{prompt}\nASSISTANT:"
            return prompt

    def generate(
        self,
        image,
        prompt,
        max_new_tokens=1500,
        do_sample=False,
        skip_special_tokens=True,
    ) -> tuple[str, float]:
        start_time = time()

        prompt = self.__validate_prompt_token(prompt)

        inputs = self.__processor(
            prompt,
            image,
            return_tensors="pt",
        ).to(0, torch.float16)

        raw_out = self.__model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
        )

        out = self.__processor.decode(
            raw_out[0], 
            skip_special_tokens=skip_special_tokens
        )

        out = out[out.index("ASSISTANT: ") + len("ASSISTANT: ") :]

        end_time = time()
        sec = end_time - start_time

        return out, sec
