import json
import os
import re

from src.helper.base import save_annotated_img
from src.config.run import RunConfig


def parse_output(
    input_text: str,
    img_id: str,
    prev_i: int = 0,
) -> list[dict[str, str]]:
    if not input_text.endswith("\n"):
        input_text += "\n"

    pattern_question = re.compile(r"Question:\s(.+?)\n")
    pattern_short_answer = re.compile(r"Short Answer:\s(.+?)\n")
    pattern_long_answer = re.compile(r"Reason:\s(.+?)\n")

    # Find matches using regular expressions
    questions = pattern_question.findall(input_text)
    short_answers = pattern_short_answer.findall(input_text)
    long_answers = pattern_long_answer.findall(input_text)

    ids = [i + prev_i for i in range(1, len(questions) + 1)]
    data = [
        {
            "id": i,
            "img_id": img_id,
            "question": q,
            "short_answer": sa,
            "reasoned_answer": la,
        }
        for i, q, sa, la in zip(ids, questions, short_answers, long_answers)
    ]

    return data

def export_result(
    data: list[dict], raw_data: str, total_sec: float, run_config: RunConfig
) -> None:
    model_config = run_config.get_model_config()
    data_config = run_config.get_data_config()
    run_params = run_config.get_run_params()

    metadata = {
        "test_name": run_config.get_test_name(),
        "model_name": model_config["name"],
        "prompt": run_config.get_prompt_type(),
        "expected_data_count": run_params["num_per_inference"] * data_config["count"],
        "generated_data_count": len(data),
        "total_sec": total_sec,
    }

    test_name = run_config.get_test_name()
    result_folder = f"./result/{test_name}"
    id = 1
    while os.path.exists(result_folder):
        result_folder = f"./result/{test_name}_{id}"
        id += 1
    os.makedirs(result_folder)

    misc_folder = os.path.join(result_folder, "misc")
    os.makedirs(misc_folder)

    metadata_path = os.path.join(result_folder, "metadata.json")
    with open(metadata_path, "w") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)

    res_path = os.path.join(result_folder, "res.json")
    with open(res_path, "w") as res_file:
        json.dump(data, res_file, indent=2)

    raw_path = os.path.join(misc_folder, "raw.txt")
    with open(raw_path, "w") as raw_file:
        raw_file.write(raw_data)

    # # If annot_metadatas is not empty, get every image using save_annotated_img to the misc/annot_img folder with name {id}_annot.jpg
    # if annot_metadatas:
    #     annot_img_folder = os.path.join(misc_folder, "annot_img")
    #     os.makedirs(annot_img_folder)
    #     for metadata in annot_metadatas:
    #         try:
    #             save_annotated_img(
    #                 metadata["complete_tensor"],
    #                 os.path.join(annot_img_folder, f"{metadata['id']}_annot.jpg"),
    #             )
    #         except:
    #             continue
    