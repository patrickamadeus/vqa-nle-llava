import json
import os
import re

from src.helper.base import save_annotated_img

def parse_output(
    input_text: str,
    img_id: str,
    prev_i: int = 0,
) -> list[dict[str, str]]:
    """
    Parse output text and extract question, short answer, and long answer.

    Args:
    - input_text (str): The input text containing question, short answer, and long answer.
    - img_id (str): The image ID associated with the output.
    - prev_i (int): The previous ID, used for generating sequential IDs.

    Returns:
    - list[dict[str, str]]: A list of dictionaries containing parsed data.
    """
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
    data: list[dict],
    total_data: int,
    total_gen: int,
    total_sec: int,
    primary_raw_out: str,
    inter_raw_out: str,
    test_name: str,
    dataset_name: str,
    model_name: str,
    model_family: str,
    prompt_primary: str,
    prompt_inter: str,
    inter_dict: dict,
    annot_metadatas: list[dict],
):
    # Create a new folder named "result" with a numbered identifier if it exists
    result_folder = f"./result/{test_name}"
    identifier = 1
    while os.path.exists(result_folder):
        result_folder = f"./result/{test_name}_{identifier}"
        identifier += 1

    os.makedirs(result_folder)
    misc_folder = os.path.join(result_folder, "misc")
    os.makedirs(misc_folder)

    # Construct metadata dictionary
    metadata = {
        "test_name": test_name,
        "dataset_name": dataset_name,
        "total_data": total_data,
        "total_gen": total_gen,
        "total_sec": total_sec,
        "model_name": model_name,
        "model_family": model_family,
        "prompt_primary": prompt_primary,
        "prompt_inter": prompt_inter,
    }

    # Export metadata to metadata.json
    metadata_path = os.path.join(result_folder, "metadata.json")
    with open(metadata_path, "w") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)

    # Export data to res.json
    res_path = os.path.join(result_folder, "res.json")
    with open(res_path, "w") as res_file:
        json.dump(data, res_file, indent=2)

    # Export inter_dict to inter.json
    inter_dict_path = os.path.join(misc_folder, "inter.json")
    with open(inter_dict_path, "w") as res_file:
        json.dump(inter_dict, res_file, indent=2)

    # Export primary_raw_out to primary.txt
    primary_path = os.path.join(misc_folder, "primary.txt")
    with open(primary_path, "w") as primary_file:
        primary_file.write(primary_raw_out)

    # Export inter_raw_out to secondary.txt
    inter_path = os.path.join(misc_folder, "inter.txt")
    with open(inter_path, "w") as inter_file:
        inter_file.write(inter_raw_out)

    # If annot_metadatas is not empty, get every image using save_annotated_img to the misc/annot_img folder with name {id}_annot.jpg
    if annot_metadatas:
        annot_img_folder = os.path.join(misc_folder, "annot_img")
        os.makedirs(annot_img_folder)
        for metadata in annot_metadatas:
            try:
                save_annotated_img(
                    metadata["complete_tensor"],
                    os.path.join(annot_img_folder, f"{metadata['id']}_annot.jpg"),
                )
            except:
                continue


def verif_digit(s: str):
    s = s.strip()
    if s.isnumeric():
        return int(s)
    return -1


def export_eval(
    test_name: str,
    eval_model: str,
    ids: [int],
    factoid_scores: [str],
    reasoning_scores: [str],
    raw_output: str,
    total_eval_time: int,
    in_token: int,
    out_token: int,
    eval_prompt_factoid: str,
    eval_prompt_reasoning: str,
) -> None:
    accs, logics, clears, details, irrels, plauss = [], [], [], [], [], []

    total_bad_tc = 0
    for ID, fs, rs in zip(ids, factoid_scores, reasoning_scores):
        fs, logic, clear, detail, irrel, plaus = (
            verif_digit(s) for s in [fs] + rs.split(";")
        )
        accs.append(fs)
        logics.append(logic)
        clears.append(clear)
        details.append(detail)
        irrels.append(irrel)
        plauss.append(plaus)

        count_bad_tc = sum(x == -1 for x in [fs, logic, clear, detail, irrel, plaus])
        if count_bad_tc:
            print(f"{count_bad_tc} bad metrics exist for id: {ID}")
            total_bad_tc += count_bad_tc

    data = [
        {
            "id": i,
            "accuracy": a,
            "logic": l,
            "clarity": c,
            "detail": d,
            "irrelevance": ir,
            "plausibility": p,
        }
        for i, a, l, c, d, ir, p in zip(
            ids, accs, logics, clears, details, irrels, plauss
        )
    ]

    result_folder = f"./result/{test_name}"
    misc_folder = f"./result/{test_name}/misc"

    misc_path = os.path.join(misc_folder, "eval_raw.txt")
    with open(misc_path, "w") as raw_eval_file:
        raw_eval_file.write(raw_output)

    eval_res_path = os.path.join(result_folder, "eval.json")
    with open(eval_res_path, "w") as json_file:
        json.dump(data, json_file, indent=2)

    metadata_path = os.path.join(result_folder, "metadata.json")
    data = ""
    with open(metadata_path, "r") as metadata_file:
        data = json.load(metadata_file)
        data["evaluator_model_name"] = eval_model
        data["total_eval_time"] = total_eval_time
        data["eval_prompt_factoid"] = eval_prompt_factoid
        data["eval_prompt_reasoning"] = eval_prompt_reasoning
        data["eval_success_rate"] = round(1 - total_bad_tc / (len(ids) * 6), 2)

    with open(metadata_path, "w") as metadata_file:
        json.dump(data, metadata_file, indent=2)

    print(f"Total bad TC metrics {total_bad_tc}/{len(ids) * 6}")
