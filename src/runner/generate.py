from tqdm import tqdm
import logging
from sentence_transformers import SentenceTransformer, util
import warnings
import re

warnings.filterwarnings("ignore")

from src.model import LVLM
from src.dataset import Dataset
from src.config.run import RunConfig
from src.helper.base import init_logging, raw_output_splitter, set_seed
from src.helper.parser import parse_output

Config = RunConfig()
SEED = Config.get_seed()

init_logging()
set_seed(SEED)


class GenerateRunner:
    def __init__(
        self, dataset: Dataset, model: LVLM, prompt: str, run_cfg: RunConfig) -> None:
        self.__data = dataset.get_data()
        self.__model = model
        self.__prompt = prompt
        self.__prompt_type = run_cfg.get_prompt_type()

        self.__is_ensemble = self.__prompt == "self_consistency"
        self.__is_using_scene_graph = dataset.is_using_scene_graph()

        if self.__is_using_scene_graph and not self.__prompt_type.startswith("nonvis"):
            print("You are using scene graph annotation without using non-visual prompt.")
            user_input = input("Enter [Y/y] to proceed: ")
            if user_input.strip().lower() != 'y':
                print("Program aborted.")
                exit()
        
        if self.__is_ensemble:
            self.__sc_model = SentenceTransformer(
                "sentence-transformers/all-mpnet-base-v2"
            )
            with open("./prompt/self_consistency/question.txt") as f1, open(
                "./prompt/self_consistency/short_answer.txt"
            ) as f2, open("./prompt/self_consistency/reasoning.txt") as f3, open(
                "./prompt/self_consistency/reasoning_cot.txt"
            ) as f4, open("./prompt/self_consistency/react.txt") as f5:
                self.__q_prompt = f1.read()
                self.__a_prompt = f2.read()
                self.__r_prompt = f3.read()
                self.__r_cot_prompt = f4.read()
                self.__r_react_prompt = f5.read()

            self.__reasoning_prompts = [
                self.__r_prompt,
                self.__r_cot_prompt,
                self.__r_react_prompt,
            ]
            self.__reasoning_prompts_max_tokens = [70, 70, 300]

    def __format_prompt(self, prompt: str, prefix: str, obj_name: str = None) -> str:
        if obj_name:
            formatted_prompt = prompt.format(prefix=prefix, obj_name=obj_name)
        else:
            formatted_prompt = prompt.format(prefix=prefix)

        return formatted_prompt

    def __validate_question(self, question: str) -> str:
        questions = question.split(",")
        if len(questions) == 1:
            return questions[0]

        prefixes = ["how", "what", "why", "who", "whose", "which", "where", "when"]

        for question in questions[1:]:
            for prefix in prefixes:
                if prefix in question:
                    return questions[0] + "?"

        if "?" not in questions[-1]:
            return ",".join(questions[:-1]) + "?"

        return ",".join(questions[:]).strip("\n")

    def __validate_short_answer(self, short_answer: str) -> str:
        return short_answer.strip("\n")

    def __validate_reason(self, reason: str) -> str:
        sentences = reason.split(".")
        last_sentence = sentences[-1]

        if not last_sentence.endswith("."):
            return ".".join(sentences[:-1]) + "."

        return ".".join(sentences[:]).strip("\n") + "."

    def __get_top_reasoning(self, reasoning_list: list[str]) -> str:
        if len(reasoning_list) == 1:
            return reasoning_list[0]

        embeddings = self.__sc_model.encode(reasoning_list, convert_to_tensor=True)
        cos_sim_matrix = util.pytorch_cos_sim(embeddings, embeddings)

        scores = cos_sim_matrix.sum(dim=1) - 1

        return reasoning_list[scores.argmax().item()]

    def __generate_question(self, img, prefix):
        raw_question = self.__model.generate(
            image=img,
            prompt=self.__q_prompt.format(prefix=prefix),
            max_new_tokens=20,
        )

        return self.__validate_question(raw_question)

    def __generate_short_answer(self, img, question):
        raw_short_answer = self.__model.generate(
            image=img,
            prompt=self.__a_prompt.format(question=question),
            max_new_tokens=25,
        )

        return self.__validate_short_answer(raw_short_answer)

    def __extract_react(self, text):
        observation_pattern = r"Observation:(.*?)Thoughts:"
        thoughts_pattern = r"Thoughts:(.*?)Action:"
        action_pattern = r"Action:(.*?)Reason:"

        observation_match = re.search(observation_pattern, text, re.DOTALL)
        thoughts_match = re.search(thoughts_pattern, text, re.DOTALL)
        action_match = re.search(action_pattern, text, re.DOTALL)

        if observation_match and thoughts_match and action_match:
            reason_pattern = r"Reason:(.*?)$"
            reason_match = re.search(
                reason_pattern, action_match.group(1).strip(), re.DOTALL
            )
            if reason_match:
                return reason_match.group(1).strip()

        return text.strip()

    def __generate_reasonings(self, img, question, short_answer):
        reasonings = []
        raw_reasonings = []

        for i, prompt in enumerate(self.__reasoning_prompts):
            raw_reason = self.__model.generate(
                image=img,
                prompt=prompt.format(question=question, short_answer=short_answer),
                max_new_tokens=self.__reasoning_prompts_max_tokens[i],
            )

            if i == len(self.__reasoning_prompts) - 1:
                raw_reason = self.__extract_react(raw_reason)

            reasonings.append(self.__validate_reason(raw_reason))
            raw_reasonings.append(raw_output_splitter(i, raw_reason))

        return reasonings, raw_reasonings

    def __generate_self_consistency_data(self, img, prefix) -> str:
        question = self.__generate_question(img, prefix)
        short_answer = self.__generate_short_answer(img, question)
        reasonings, raw_reasonings = self.__generate_reasonings(
            img, question, short_answer
        )
        top_reasoning = self.__get_top_reasoning(reasonings)

        raw_data = f"Question: {question}\nShort Answer: {short_answer}\nrReason: {top_reasoning}\n"

        return raw_data

    def generate_synthetic_data(self):
        logging.info("Starting Generation...")

        total_data = []
        prev_id = 0
        raw_output = ""

        for img_id, img, prefix, obj_name in tqdm(self.__data):
            if not self.__is_ensemble:
                raw_data = self.__model.generate(
                    image=img,
                    prompt=self.__format_prompt(self.__prompt, prefix, obj_name),
                )
            else:
                raw_data = self.__generate_self_consistency_data(img, prefix)

            parsed_data = parse_output(raw_data, img_id, prev_i=prev_id)

            total_data.extend(parsed_data)

            prev_id += len(parsed_data)
            raw_output += raw_output_splitter(img_id, f"{prefix}\n\n" + raw_data)

            logging.info(
                f"[{img_id}] - success generated {len(parsed_data)} synthetic data(s)"
            )

        return total_data, raw_output
