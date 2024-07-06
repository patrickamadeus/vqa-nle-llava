import base64
import json
import logging
import os
import random

import numpy as np
import torch
import torchvision
from yaml import safe_load


def load_config(config_path: str) -> dict:
    with open(config_path) as file_path:
        config = safe_load(file_path)
    return config


def init_logging(
    level=logging.INFO,
    save_to_file=False,
    formatter="%(asctime)s-%(levelname)s-%(message)s",
):
    logging.basicConfig(
        #         filename = "test.log",
        level=level,
        format=formatter,
        datefmt="%d-%b-%y %H:%M:%S",
    )


def set_seed(seed: int) -> None:
    """
    Set the seed for random number generators for reproducibility.

    Args:
    - seed (int): The seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def format_timediff(
    seconds: int, format_str: str = "{hours}h{minutes}m{seconds}"
) -> str:
    """
    Format the time difference in seconds into a readable string.

    Args:
    - seconds (int): The time difference in seconds.
    - format_str (str): The format string for the output.

    Returns:
    - str: The formatted time difference string.
    """
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    formatted_timediff = format_str.format(
        hours=hours, minutes=minutes, seconds=seconds
    )

    return formatted_timediff


def get_all_filepaths(folder_path: str, n=99999999) -> tuple[list[str], int]:
    """
    Get all file paths from the specified folder.

    Args:
    - folder_path (str): The path to the folder.
    - n (int): Maximum number of file paths to retrieve (default is 99999999).

    Returns:
    - tuple[list[str], int]: A tuple containing a list of file paths and the size of the list.
    """
    file_paths = [
        os.path.join(root, file)
        for root, dirs, files in os.walk(folder_path)
        for file in files
        if os.path.basename(root) == os.path.basename(folder_path)
    ][:n]

    return file_paths, len(file_paths)


def get_filepaths_iterator(folder_path: str, n: int):
    """
    Get an iterator of file paths from the specified folder up to a limit of n files.

    Args:
    - folder_path (str): The path to the folder.
    - n (int): The maximum number of files to include in the iterator.

    Returns:
    - Iterator[str]: An iterator of file paths.
    """
    file_paths = (
        os.path.join(root, file)
        for root, dirs, files in os.walk(folder_path)
        for file in files
    )
    limited_file_paths = (file_path for _, file_path in zip(range(n), file_paths))
    return limited_file_paths


def get_filename(long_path: str) -> str:
    """
    Get the filename from a long path.

    Args:
    - long_path (str): The long path containing the filename.
    - extension (bool): Whether include extension / no

    Returns:
    - str: The filename.
    """
    filename = os.path.basename(long_path)
    raw_filename = os.path.splitext(filename)[0]

    return filename, raw_filename


def get_all_valid_filepaths(
    folder_path: str, scene_graph: dict, n=99999999
) -> tuple[list[str], int]:
    """
    Get all file paths from the specified folder.

    Args:
    - folder_path (str): The path to the folder.
    - n (int): Maximum number of file paths to retrieve (default is 99999999).

    Returns:
    - tuple[list[str], int]: A tuple containing a list of file paths and the size of the list.
    """
    keys = set(scene_graph.keys())
    file_paths = [
        os.path.join(root, file)
        for root, dirs, files in os.walk(folder_path)
        for file in files
        if (os.path.basename(root) == os.path.basename(folder_path))
        and (get_filename(file)[-1] in keys)
    ][:n]

    return file_paths, len(file_paths)


def unpack_json(json_file_path):
    try:
        with open(json_file_path, "r") as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: File '{json_file_path}' not found.")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON in '{json_file_path}': {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def annotate_images(img_path, graph, num_obj=5, min_area_div=100):  ### PINDAH KE PREPROC
    COLORS = [
        "red",
        "green",
        "blue",
        "yellow",
        "purple",
        "orange",
        "brown",
        "pink",
        "gray",
        "cyan",
    ]
    img = torchvision.io.read_image(img_path)

    img_area = img.size()[-1] * img.size()[-2]
    annotated_imgs = []
    bboxs = []

    # Create num_obj images with only one annotation
    for v in list(graph["objects"].values()):
        x, y, w, h = v["x"], v["y"], v["w"], v["h"]
        if w * h * min_area_div < img_area:
            continue

        bbox = [x, y, x + w, y + h]
        bboxs.append(bbox)
        bbox = torch.tensor([bbox])

        img_tensor = torchvision.utils.draw_bounding_boxes(
            img, bbox, width=3, colors=["red"]
        )
        img_pil = torchvision.transforms.ToPILImage()(img_tensor)

        name = v["name"]
        annotated_imgs.append((name, img_pil))

        num_obj -= 1
        if num_obj == 0:
            break

    # Draw all annotations on the image
    complete_annot_img_tensor = torchvision.utils.draw_bounding_boxes(
        img, torch.tensor(bboxs), width=3, colors=COLORS[: len(bboxs)]
    )

    return annotated_imgs, complete_annot_img_tensor


def save_annotated_img(tensor, path):
    img_pil = torchvision.transforms.ToPILImage()(tensor)
    img_pil.save(path)


def raw_output_splitter(out_id, out_content, extras=None): ### PINDAH KE PREPROC/ POSTPROC
    if out_content != "":
        out = f"{out_id}\n-------------------------------\n{out_content}\n\n"
        if extras is not None:
            out += '\n'.join([f'Choice {i + 1}: {extras[i]}' for i in range(len(extras))])
        return out + "\n\n"
    return ""


def expand_prefix_stratify(prefixes, props, total_length): ### PINDAH KEINFERENCE
    props = [(total_length * prop) // sum(props) for prop in props]

    expanded_list = []
    for i, prop in enumerate(props):
        expanded_list += [prefixes[i]] * prop

    expanded_list += [prefixes[0]] * (total_length - sum(props))
    random.shuffle(expanded_list)

    return expanded_list[:total_length]


def validate_question(q): ### PINDAH KE POSTPROC
    questions = q.split(",")
    if len(questions) == 1:
        return questions[0]
    
    prefixes = ["how", "what", "why", "who", "whose", "which", "where","when"]
    proportion = [2,2,2,1,1,1,2,1,2,1,]
    
    for question in questions[1:]:
        for prefix in prefixes:
            if prefix in question:
                return questions[0] + "?"
    
    if "?" not in questions[-1]:
        return ','.join(questions[:-1]) + "?"

    return ','.join(questions[:]).strip("\n")

def validate_short_answer(a): ### PINDAH KE POSTPROC
    return a.strip("\n")


def validate_reason(s): ### PINDAH KE POSTPROC
    sentences = s.split('.')
    last_sentence = sentences[-1]
    
    if not last_sentence.endswith('.'):
        return '.'.join(sentences[:-1]) + "."
    
    return '.'.join(sentences[:]).strip("\n") + "."