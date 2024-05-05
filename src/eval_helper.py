import base64
import io
import json
import os
import sys
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import table
import seaborn as sns
from irrCAC.raw import CAC
from PIL import Image
import plotly.graph_objects as go

#----
#----

from src.base import load_config, unpack_json

METRICS = ["accuracy", "logic", "clarity", "detail", "irrelevance", "plausibility"]

cfg = load_config("./", "eval.yml")

SEED = cfg["seed"]
TEST_NAME = cfg["test_name"]
EVAL_NUM = cfg["eval_amount"]

MULTI_RESULT_PATH = f"./result/multieval/{TEST_NAME}/"
SINGLETON_RESULT_PATH = "./result/{test_name}/eval/"
RULES_PATH = "./prompt/eval/rules.txt"

MAP_INDEX = {
    "10-4_vicuna13-vip_nonvis-optim_500" : "13b_nonvis",
    "10-4_vicuna7_naive-optim_500": "7b_naive",
    "10-4_vicuna13_naive-optim_500": "13b_naive",
    "10-4_vicuna13_qg-story-optim_500": "13b_qg-story",
}


class NullWriter(object):
    def write(self, arg):
        pass


nullwrite = NullWriter()
oldstdout = sys.stdout


def load_eval_res(path, mode="csv", sep=","):
    if mode == "json":
        with open(path, "r") as f:
            data = json.load(f)
            df = pd.DataFrame(data)
    elif mode == "csv":
        df = pd.read_csv(path, sep=sep)

    df.replace(-1, np.nan, inplace=True)
    return df


def export_eval(name, data, test_name=None, mode="json"):
    path = SINGLETON_RESULT_PATH.format(test_name=test_name)
    if not os.path.exists(path):
        os.makedirs(path)

    if mode == "json":
        with open(path + f"{name}.json", "w") as json_file:
            json.dump(data, json_file)

    elif mode == "plt":
        image_bytes = base64.b64decode(data)

        image = Image.open(io.BytesIO(image_bytes))

        image_path = os.path.join(path, f"{name}.jpg")
        image.save(image_path)

    elif mode == "xlsx":
        with open("./prompt/eval/rules.txt", "r") as rule_f:
            rule_list = rule_f.readlines()

        rules = pd.DataFrame({"rules": rule_list})

        writer = pd.ExcelWriter(path + f"scoring_template.xlsx", engine="xlsxwriter")
        rules.to_excel(writer, sheet_name="rules")
        data.to_excel(writer, sheet_name="scoresheet")
        writer.close()
    
    elif mode == "df":

        # fig, ax = plt.subplots(figsize=(10, 2)) # set size frame
        # ax.xaxis.set_visible(False)  # hide the x axis
        # ax.yaxis.set_visible(False)  # hide the y axis
        # ax.set_frame_on(False)  # no visible frame, uncomment if size is ok
        # tabla = table(ax, data, loc='upper right', colWidths=[0.17]*len(data.columns))  # where df is your data frame
        # tabla.auto_set_font_size(False) # Activate set fontsize manually
        # tabla.set_fontsize(12) # if ++fontsize is necessary ++colWidths
        # tabla.scale(1.2, 1.2) # change size table
        # plt.savefig('table.png', transparent=True)

        # fig = go.Figure(data=[
        #                     go.Table(
        #                         header=dict(values=list(data.columns),align='center'),
        #                         cells=dict(values=data.values,
        #                                 fill_color = [["white","lightgrey"]*data.shape[0]],
        #                                 align='center'
        #                                 )
        #                             )
        #                     ])
    
        if not os.path.exists(MULTI_RESULT_PATH):
            os.makedirs(MULTI_RESULT_PATH)
        
        # also export the data as csv
        data.to_csv(os.path.join(MULTI_RESULT_PATH, f"{name}.csv"))

        # also export the data as xlsx
        data.to_excel(os.path.join(MULTI_RESULT_PATH, f"{name}.xlsx"))

        # filename = os.path.join(MULTI_RESULT_PATH, f"{name}.jpg")
        # fig.write_image(filename,scale=6)
        # plt.savefig(filename, transparent=True)
        


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

    common_ids_set = set(dfs[0]["id"].values)
    for df in dfs[1:]:
        common_ids_set &= set(df["id"].values)

    return common_ids_set


def dfs_to_CAC(dfs):
    METRICS = ["accuracy", "logical", "clarity", "detail", "irrelevancy"]
    CACs = {}

    common_ids_set = common_ids(dfs)

    for metric in METRICS:
        series = [df[df["id"].isin(common_ids_set)][metric] for df in dfs]
        cols = [f"rater_{i}" for i in range(1, len(dfs) + 1)]
        CACs[metric] = pd.concat(series, keys=cols, axis=1)

    return CACs


def gwet_AC2(test_name, weights="ordinal"):
    transformed_dfs = transform_raw_to_dfs(test_name)
    cleaned_transformed_dfs = [get_clean_df(df, mode = "remove")[0] for df in transformed_dfs]
    
    if len(cleaned_transformed_dfs) == 1:
        return {
            "accuracy": 1.0,
            "logical": 1.0,
            "clarity": 1.0,
            "detail": 1.0,
            "irrelevancy": 1.0,
            "overall": 1.0
        }

    cac_by_metric_dict = dfs_to_CAC(cleaned_transformed_dfs)
    gwet_by_metric_dict = {}

    scores = []
    for metric, cac in cac_by_metric_dict.items():
        cac_metric = CAC(cac, weights=weights)

        score = cac_metric.gwet()["est"]["coefficient_value"]
        gwet_by_metric_dict[metric] = score
        scores.append(score)

    gwet_by_metric_dict["overall"] = sum(scores) / len(scores)

    return gwet_by_metric_dict


def gen_size_hist(test_name, all=True):
    data = unpack_json(f"./result/{test_name}/res.json")[:EVAL_NUM]
    
    if not all:
        clean_ids = set(get_subj_mutual_data(test_name, mode="remove")[0][0]["id"].values)
        data = [item for item in data if item["id"] in clean_ids]

    question_lengths = [len(item["question"].split()) for item in data]
    short_answer_lengths = [len(item["short_answer"].split()) for item in data]
    reasoned_answer_lengths = [len(item["reasoned_answer"].split()) for item in data]

    # Create subplots with Seaborn
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

    # Plot histograms
    sns.histplot(question_lengths, ax=axes[0], kde=True)
    sns.histplot(short_answer_lengths, ax=axes[1], kde=True)
    sns.histplot(reasoned_answer_lengths, ax=axes[2], kde=True)

    # Set the x-axis limits for each plot
    axes[0].set_xlim(0, 20)
    axes[1].set_xlim(0, 50)
    axes[2].set_xlim(0, 100)

    # Set titles
    axes[0].set_title("Question Word Length")
    axes[1].set_title("Short Answer Word Length")
    axes[2].set_title("Reasoned Answer Word Length")

    # Set common y-label
    fig.text(0.04, 0.5, "Frequency", va="center", rotation="vertical")

    # Set common suptitle
    fig.suptitle(f"Word Length Histograms - {test_name}", fontsize=16)

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


def gen_question_prefix(test_name, all = True):
    data = unpack_json(f"./result/{test_name}/res.json")[:EVAL_NUM]

    if not all:
        clean_ids = set(get_subj_mutual_data(test_name, mode="remove")[0][0]["id"].values)
        data = [item for item in data if item["id"] in clean_ids]

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
    plt.title(f"Question Prefix Distribution - {test_name}")

    # Save plot as bytes in memory
    buf = io.BytesIO()
    plt.savefig(buf, format="jpg")
    buf.seek(0)

    # Convert bytes to base64 encoded string
    image_base64 = base64.b64encode(buf.read()).decode("utf-8")

    # Close plot to free memory
    plt.close()

    return image_base64


def gen_subjective_xlsx(test_name):
    data = unpack_json(f"./result/{test_name}/res.json")[:EVAL_NUM]
    cols = ["id", "img_id", "question", "short_answer", "reasoned_answer"]
    df = pd.DataFrame(data, columns=cols)

    evaluation_criteria = [
        "accuracy",
        "logical",
        "clarity",
        "detail",
        "irrelevancy",
        "plausibility",
    ]
    for criterion in evaluation_criteria:
        df[criterion] = ""

    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    sample_df = df.head(100)

    return sample_df[cols[0:1] + evaluation_criteria + cols[1:]]


# Subjective Evaluation Helper Data Preprocessing Function
def transform_raw_to_dfs(
    test_name: str, sheet_name: str = "scoresheet"
) -> List[pd.DataFrame]:
    xlsx_dir = f"./result/{test_name}/eval/xlsx/"
    dfs = []

    for file in os.listdir(xlsx_dir):
        filename, ext = os.path.splitext(file)
        evaluator = filename.split("_")[-1]

        if ext == ".xlsx":
            df = pd.read_excel(xlsx_dir + file, sheet_name=sheet_name)[
                [
                    "id",
                    "img_id",
                    "accuracy",
                    "logical",
                    "clarity",
                    "detail",
                    "irrelevancy",
                ]
            ]
        elif ext == ".csv":
            df = pd.read_csv(xlsx_dir + file)[
                [
                    "id",
                    "img_id",
                    "accuracy",
                    "logical",
                    "clarity",
                    "detail",
                    "irrelevancy",
                ]
            ]

        df["evaluator"] = evaluator
        dfs.append(df)

    return dfs


def get_clean_df(df: pd.DataFrame, mode = "remove") -> Tuple[pd.DataFrame, float]:
    df.dropna(subset=["accuracy", "logical", "clarity", "detail", "irrelevancy"], inplace=True)

    clean_rate = None
    if mode == "remove":
        clean_df = df[(df != -1.0).all(axis=1)]
        clean_rate = len(clean_df) / len(df)
    elif mode == "replace":
        temp_clean_df = df[(df != -1.0).all(axis=1)]
        clean_df = df.replace(-1.0, 1.0)
        clean_rate = len(temp_clean_df) / len(df)

    return clean_df, clean_rate


def get_clean_ids(df: pd.DataFrame) -> set:
    return set(get_clean_df(df)[0]["id"].values)


def get_subj_mutual_data(test_name: str, mode = "remove") -> Tuple[List[pd.DataFrame], List[float]]:
    transformed_dfs = transform_raw_to_dfs(test_name)
    cleaned_transformed_dfs = [get_clean_df(df, mode=mode) for df in transformed_dfs]

    cleaned_dfs = [df for df, _ in cleaned_transformed_dfs]
    mutual_clean_rate = [rate for _, rate in cleaned_transformed_dfs]

    common_ids_set = common_ids(cleaned_dfs)
    mutual_dfs = [df[df["id"].isin(common_ids_set)] for df in cleaned_dfs]

    return mutual_dfs, mutual_clean_rate


def gen_subjective_quant_analysis(test_names: List[str]) -> dict:
    res = {}

    for test_name in test_names:
        
        mutual_dfs, mutual_clean_rate = get_subj_mutual_data(test_name=test_name, mode = "replace")

        metric = ["accuracy", "logical", "clarity", "detail", "irrelevancy"]
        mean_scores = [df[metric].mean() for df in mutual_dfs]
        ovr_mean_scores = pd.concat(mean_scores, axis=1).mean(axis=1)
        ovr_std_scores = pd.concat(mean_scores, axis=1).std(axis=1)

        mean_scores = [mean_scores[i].to_dict() for i in range(len(mean_scores))]
        
        gwet_ac2 = gwet_AC2(test_name)

        res[test_name] = {
            "amount": len(mutual_dfs[0]),
            "gen_rate": np.mean(mutual_clean_rate),
            "ovr_mean_scores": ovr_mean_scores.to_dict(),
            "ovr_std_scores": ovr_std_scores.to_dict(),
            "ovr_gwet_ac2": gwet_ac2["overall"],
            
            "mean_scores_per_sample": mean_scores,
            "gwet_per_metrics": gwet_ac2,
        }

    return res


def gen_quant_subj_df(
    test_names: List[str], export_detail: bool = False
) -> pd.DataFrame:
    
    quant_subj_res = gen_subjective_quant_analysis(test_names)
    res = {}
    
    for test_name, data in quant_subj_res.items():
        if export_detail:
            with open(f"./result/{test_name}/eval/quant_subj.json", "w") as f:
                json.dump(quant_subj_res[test_name], f)

        res[test_name] = {
            "amount": data["amount"],
            "gen_rate": data["gen_rate"],
            "gwet_ac2": data["ovr_gwet_ac2"],
            **{
                f"avg_{metric}": data["ovr_mean_scores"][metric]
                for metric in data["ovr_mean_scores"].keys()
            },
            **{
                f"std_{metric}": data["ovr_std_scores"][metric]
                for metric in data["ovr_std_scores"].keys()
            },
        }
    
    df = round(pd.DataFrame(res).T, 2)
    if MAP_INDEX:
        df.rename(index=MAP_INDEX, inplace=True)
    
    return df


def gen_subj_rank(test_names: List[str]) -> pd.DataFrame:
    quant_subj_res = gen_subjective_quant_analysis(test_names)
    res = {}
    for test_name, data in quant_subj_res.items():
        content = data["ovr_mean_scores"]
        content["amount"] = data["amount"]
        content["gen_rate"] = data["gen_rate"]
        content["gwet_ac2"] = data["ovr_gwet_ac2"]
        res[test_name] = content

    df = pd.DataFrame(res).T
    if MAP_INDEX:
        df.rename(index=MAP_INDEX, inplace=True)

    rank_df = pd.DataFrame(index=[i + 1 for i in range(len(df.index))], columns=df.columns)
    
    for col in df.columns:
        asc = True if col == "irrelevancy" else False
        ranked_idx = df.sort_values(by=col, ascending=asc)[col].index
        ranked_val = np.round(df.sort_values(by=col, ascending=asc)[col].values, 2)

        rank_df[col] = [f"{ranked_idx[i]} - ({ranked_val[i]})" for i in range(len(ranked_idx))]

    return rank_df
