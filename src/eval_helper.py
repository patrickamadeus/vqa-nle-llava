from src.base import load_config, unpack_json
from irrCAC.raw import CAC
import json
import os
import io
import base64
from PIL import Image
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

METRICS = ["accuracy", "logic", "clarity", "detail", "irrelevance", "plausibility"]

cfg = load_config("./", "eval.yml")
TEST_NAME = cfg["test_name"]
RESULT_PATH = f"./result/{TEST_NAME}/eval/"


def load_eval_res(path, mode="csv", sep=","):
    if mode == "json":
        with open(path, "r") as f:
            data = json.load(f)
            df = pd.DataFrame(data)
    elif mode == "csv":
        df = pd.read_csv(path, sep=sep)

    df.replace(-1, np.nan, inplace=True)
    return df


def export_eval(name, data, mode="json", RESULT_PATH=RESULT_PATH):
    if not os.path.exists(RESULT_PATH):
        os.makedirs(RESULT_PATH)

    if mode == "json":
        with open(RESULT_PATH + f"{name}.json", "w") as json_file:
            json.dump(data, json_file)

    elif mode == "plt":
        image_bytes = base64.b64decode(data)

        image = Image.open(io.BytesIO(image_bytes))

        image_path = os.path.join(RESULT_PATH, f"{name}.jpg")
        image.save(image_path)


def transform_list_to_dfs(test_name, mode="csv", sep=";"):
    # Define the directory where the CSV files are located
    directory = f"./result/{test_name}/misc/"

    # Get all .csv files in the directory
    file_paths = [
        os.path.join(directory, filename)
        for filename in os.listdir(directory)
        if filename.endswith(".csv")
    ]

    # Load each CSV file into a DataFrame
    dataframes = [load_eval_res(path, mode=mode, sep=sep) for path in file_paths]

    return dataframes


def bulk_transform_dfs(df_list):
    """
    Perform bulk transformations on a list of DataFrames.

    1. Transform column names.
    2. Transform dtype to int.
    3. Extract only needed columns.

    Parameters:
    - df_list (list): List of DataFrames to be transformed.

    Returns:
    - list: List of transformed DataFrames.
    """
    transformed_dfs = []

    for df in df_list:
        # Task 1: Transform column names
        df = df.rename(
            columns={
                "factoid": "accuracy",
                "r0": "logic",
                "r1": "clarity",
                "r2": "detail",
                "r3": "irrelevance",
                "r4": "plausibility",
            }
        )

        needed_columns = [
            "id",
            "accuracy",
            "logic",
            "clarity",
            "detail",
            "irrelevance",
            "plausibility",
        ]
        df = df[needed_columns]

        transformed_dfs.append(df)

    return transformed_dfs


def common_ids(dfs):
    if not dfs:
        return set()

    common_ids_set = set(dfs[0]["id"])
    for df in dfs[1:]:
        common_ids_set &= set(df["id"])

    return common_ids_set


def dfs_to_CAC(dfs):
    METRICS = ["accuracy", "logic", "clarity", "detail", "irrelevance", "plausibility"]
    CACs = {}

    common_ids_set = common_ids(dfs)

    for metric in METRICS:
        series = [df[df["id"].isin(common_ids_set)][metric] for df in dfs]
        cols = [f"rater_{i}" for i in range(1, len(dfs) + 1)]
        CACs[metric] = pd.concat(series, keys=cols, axis=1)

    return CACs


def gwet_AC2(TEST_NAME, weights="ordinal"):
    dfs = transform_list_to_dfs(TEST_NAME)
    transformed_dfs = bulk_transform_dfs(dfs)

    cac_by_metric_dict = dfs_to_CAC(transformed_dfs)
    gwet_by_metric_dict = {}

    scores = []
    for metric, cac in cac_by_metric_dict.items():
        cac_metric = CAC(cac, weights=weights)

        score = cac_metric.gwet()["est"]["coefficient_value"]
        gwet_by_metric_dict[metric] = score
        scores.append(score)

    gwet_by_metric_dict["overall"] = sum(scores) / len(scores)

    return gwet_by_metric_dict


def gen_size_hist(TEST_NAME):
    data = unpack_json(f"./result/{TEST_NAME}/res.json")
    question_lengths = [len(item["question"].split()) for item in data]
    short_answer_lengths = [len(item["short_answer"].split()) for item in data]
    reasoned_answer_lengths = [len(item["reasoned_answer"].split()) for item in data]

    # Create subplots with Seaborn
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

    # Plot histograms
    sns.histplot(question_lengths, ax=axes[0], kde=True)
    sns.histplot(short_answer_lengths, ax=axes[1], kde=True)
    sns.histplot(reasoned_answer_lengths, ax=axes[2], kde=True)

    # Set titles
    axes[0].set_title("Question Word Length")
    axes[1].set_title("Short Answer Word Length")
    axes[2].set_title("Reasoned Answer Word Length")

    # Set common y-label
    fig.text(0.04, 0.5, "Frequency", va="center", rotation="vertical")

    # Set common suptitle
    fig.suptitle(f"Word Length Histograms - {TEST_NAME}", fontsize=16)

    # Adjust layout
    plt.tight_layout()

    # Save plot as bytes in memory
    buf = io.BytesIO()
    plt.savefig(buf, format="jpg")
    buf.seek(0)

    # Convert bytes to base64 encoded string
    image_base64 = base64.b64encode(buf.read()).decode("utf-8")

    # Close plot to free memory
    plt.close()

    return image_base64


def gen_question_prefix(TEST_NAME):
    data = unpack_json(f"./result/{TEST_NAME}/res.json")
    question_prefixes = {}

    # Count occurrences of question prefixes
    for item in data:
        prefix = item["question"].split()[0]  # Assuming the prefix is the first word
        question_prefixes[prefix] = question_prefixes.get(prefix, 0) + 1

    # Create a pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(
        question_prefixes.values(),
        labels=question_prefixes.keys(),
        autopct="%1.1f%%",
        startangle=140,
    )

    # Set title
    plt.title(f"Question Prefix Distribution - {TEST_NAME}")

    # Save plot as bytes in memory
    buf = io.BytesIO()
    plt.savefig(buf, format="jpg")
    buf.seek(0)

    # Convert bytes to base64 encoded string
    image_base64 = base64.b64encode(buf.read()).decode("utf-8")

    # Close plot to free memory
    plt.close()

    return image_base64
