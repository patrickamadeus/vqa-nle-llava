import base64
import io
import json
import os
import sys
from typing import List, Tuple
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from irrCAC.raw import CAC
from PIL import Image
from scipy.stats import pearsonr
from scipy.spatial.distance import jensenshannon

from src.helper.base import load_config, unpack_json

METRICS = ["accuracy", "logic", "clarity", "detail", "irrelevance", "plausibility"]

cfg = load_config("./eval.yml")

SEED = cfg["seed"]
TEST_NAME = cfg["eval_name"]
EVAL_NUM = cfg["eval_amount"]

MULTI_RESULT_PATH = f"./result/multieval/{TEST_NAME}/"
SINGLETON_RESULT_PATH = "./result/{test_name}/eval/"
RULES_PATH = "./prompt/eval/rules.txt"

MAP_INDEX = {
    "10-4_vicuna13-vip_nonvis-optim_500" : "13b_nonvis",
    "10-4_vicuna7_naive-optim_500": "7b_naive",
    "10-4_vicuna13_naive-optim_500": "13b_naive",
    "10-4_vicuna13_qg-story-optim_500": "13b_qg-story",
    "vicuna13_1.6_self-const-limit": "13b_self-const"
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
    
        if not os.path.exists(MULTI_RESULT_PATH):
            os.makedirs(MULTI_RESULT_PATH)
        
        # also export the data as csv
        data.to_csv(os.path.join(MULTI_RESULT_PATH, f"{name}.csv"))

        # also export the data as xlsx
        data.to_excel(os.path.join(MULTI_RESULT_PATH, f"{name}.xlsx"))


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


def gen_size_hist(test_name, color="blue", all=True):
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
    sns.histplot(question_lengths, ax=axes[0], kde=True, color=color)
    sns.histplot(short_answer_lengths, ax=axes[1], kde=True, color=color)
    sns.histplot(reasoned_answer_lengths, ax=axes[2], kde=True, color=color)

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


def gen_dist_analysis(test_names, output_dir=MULTI_RESULT_PATH, all=False):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open("./result/human/res.json") as f:
        d = json.load(f)
    
    q_bins = np.arange(0, 21, 1.25)
    a_bins = np.arange(0, 26, 1.25)
    r_bins = np.arange(0, 51, 2.5)  

    data = d

    # Create subplots with Seaborn for human data
    fig_h, axes_h = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    fig_h.patch.set_facecolor('white')
    
    # Create histograms using seaborn for human data
    hist_q_human = sns.histplot([len(item["question"].split()) for item in data], ax=axes_h[0], bins=q_bins, color="red")
    hist_a_human = sns.histplot([len(item["short_answer"].split()) for item in data], ax=axes_h[1], bins=a_bins, color="red")
    hist_r_human = sns.histplot([len(item["reasoned_answer"].split()) for item in data], ax=axes_h[2], bins=r_bins, color="red")
    
    # Set titles for human data histograms
    axes_h[0].set_title("Question Word Length (Human)")
    axes_h[1].set_title("Short Answer Word Length (Human)")
    axes_h[2].set_title("Reasoned Answer Word Length (Human)")
    fig_h.suptitle("Human", y=1.05)
    
    # Save human data figure to file
    human_img_path = os.path.join(output_dir, "human_histograms.png")
    fig_h.savefig(human_img_path, bbox_inches='tight')
    plt.close(fig_h)
    
    res = []
    img_paths = [human_img_path]
    for sample_name in test_names:
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
        fig.patch.set_facecolor('white')

        with open(f"./result/{sample_name}/res.json") as f:
            d = json.load(f)
        
        data = d
        if not all:
            clean_ids = set(get_subj_mutual_data(sample_name, mode="remove")[0][0]["id"].values)
            data = [item for item in data if item["id"] in clean_ids]  

        hist_q = sns.histplot([len(item["question"].split()) for item in data], ax=axes[0], bins=q_bins)
        hist_a = sns.histplot([len(item["short_answer"].split()) for item in data], ax=axes[1], bins=a_bins)
        hist_r = sns.histplot([len(item["reasoned_answer"].split()) for item in data], ax=axes[2], bins=r_bins)
    
        # Set titles for sample data histograms
        axes[0].set_title("Question Word Length")
        axes[1].set_title("Short Answer Word Length")
        axes[2].set_title("Reasoned Answer Word Length")
        fig.suptitle(sample_name, y=1.05)
        
        sample_img_path = os.path.join(output_dir, f"{sample_name}_histograms.png")
        fig.savefig(sample_img_path, bbox_inches='tight')
        img_paths.append(sample_img_path)
        plt.close(fig)
    
        # Extract histogram data
        heights_q_human = [patch.get_height() for patch in hist_q_human.patches]
        heights_a_human = [patch.get_height() for patch in hist_a_human.patches]
        heights_r_human = [patch.get_height() for patch in hist_r_human.patches]
        
        heights_q_sample = [patch.get_height() for patch in hist_q.patches]
        heights_a_sample = [patch.get_height() for patch in hist_a.patches]
        heights_r_sample = [patch.get_height() for patch in hist_r.patches]
        
        # Add a small constant to avoid zeros
        epsilon = 1e-10
        heights_q_human = np.array(heights_q_human)
        heights_a_human = np.array(heights_a_human)
        heights_r_human = np.array(heights_r_human)
        
        heights_q_sample = np.array(heights_q_sample)
        heights_a_sample = np.array(heights_a_sample)
        heights_r_sample = np.array(heights_r_sample)
        
        pearson_corr_q, _ = pearsonr(heights_q_human, heights_q_sample)
        pearson_corr_a, _ = pearsonr(heights_a_human, heights_a_sample)
        pearson_corr_r, _ = pearsonr(heights_r_human, heights_r_sample)
        
        js_divergence_q = jensenshannon(heights_q_human + epsilon, heights_q_sample + epsilon)
        js_divergence_a = jensenshannon(heights_a_human + epsilon, heights_a_sample + epsilon)
        js_divergence_r = jensenshannon(heights_r_human + epsilon, heights_r_sample + epsilon)
        
        
        content = {
            "experiment": sample_name,
            "pearson_q": pearson_corr_q,
            "pearson_a": pearson_corr_a,
            "pearson_r": pearson_corr_r,
            "js_q": js_divergence_q,
            "js_a": js_divergence_a,
            "js_r": js_divergence_r
        }
        res.append(content)
    
    # Concatenate all saved images vertically using numpy
    images = [Image.open(img_path) for img_path in img_paths]
    min_width = min(img.size[0] for img in images)
    resized_images = [img.resize((min_width, int(img.size[1] * min_width / img.size[0]))) for img in images]
    image_arrays = [np.array(img) for img in resized_images]
    concatenated_image_array = np.vstack(image_arrays)
    concatenated_image = Image.fromarray(concatenated_image_array)
    
    concatenated_img_path = os.path.join(output_dir, "complete_histograms.png")
    concatenated_image.save(concatenated_img_path)
    
    res_df = pd.DataFrame(res)
    res_df.to_csv(os.path.join(output_dir, "hist_similarity.csv"), index=False)
    
    return pd.DataFrame(res), concatenated_img_path


def gen_prefix_analysis(test_names, output_dir=MULTI_RESULT_PATH, all=False):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Gather question prefixes from human data
    with open("./result/human/res.json") as f:
        human_data = json.load(f)
    
    human_prefixes = [item["question"].split()[0] for item in human_data]
    prefix_counter = Counter(human_prefixes)

    # Gather question prefixes from sample data
    ii = 0
    for sample_name in test_names:
        with open(f"./result/{sample_name}/res.json") as f:
            sample_data = json.load(f)
        print(sample_name, ii)
        if not all:
            clean_ids = set(get_subj_mutual_data(sample_name, mode="remove")[0][0]["id"].values)
            sample_data = [item for item in sample_data if item["id"] in clean_ids]
        ii += 1
        
        sample_prefixes = [item["question"].split()[0] for item in sample_data]
        prefix_counter.update(sample_prefixes)

    # Create a unique color mapping for each prefix
    unique_prefixes = list(prefix_counter.keys())
    colors = sns.color_palette("hls", len(unique_prefixes))
    color_mapping = {prefix: color for prefix, color in zip(unique_prefixes, colors)}

    # Number of datasets (human + samples)
    num_datasets = len(test_names) + 1

    # Determine the number of rows needed for the subplots (2 columns)
    num_rows = (num_datasets + 1) // 2

    # Plot the prefixes in pie charts, ordered by the total count across all tests
    fig, axes = plt.subplots(nrows=num_rows, ncols=2, figsize=(15, 5 * num_rows))
    fig.patch.set_facecolor('white')
    axes = axes.flatten()

    all_data = [("Human", human_prefixes)]
    ii = 0
    for sample_name in test_names:
        
        with open(f"./result/{sample_name}/res.json") as f:
            sample_data = json.load(f)
        if not all:
            clean_ids = set(get_subj_mutual_data(sample_name, mode="remove")[0][0]["id"].values)
            sample_data = [item for item in sample_data if item["id"] in clean_ids]
        ii += 1
        sample_prefixes = [item["question"].split()[0] for item in sample_data]
        all_data.append((sample_name, sample_prefixes))

    for i, (name, prefixes) in enumerate(all_data):
        prefix_counts = Counter(prefixes)
        sorted_prefixes = sorted(prefix_counts.items(), key=lambda x: (-prefix_counter[x[0]], -x[1]))

        labels, sizes = zip(*sorted_prefixes)
        colors = [color_mapping[label] for label in labels]
        
        axes[i].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
        axes[i].set_title(f"{name} Question Prefix Distribution")
        axes[i].set_facecolor('white')

    # Hide any unused axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    pie_chart_path = os.path.join(output_dir, "prefix_charts.jpg")
    plt.savefig(pie_chart_path)
    plt.close(fig)

    return pie_chart_path


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
        if file == ".ipynb_checkpoints":
            continue
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
    clean_df = df[(df != -1.0).all(axis=1)]
    clean_rate = len(clean_df) / len(df)
    
    return clean_df, clean_rate, len(clean_df), len(df)


def get_clean_ids(df: pd.DataFrame) -> set:
    return set(get_clean_df(df)[0]["id"].values)


def get_subj_mutual_data(test_name: str, mode = "remove") -> Tuple[List[pd.DataFrame], List[float]]:
    transformed_dfs = transform_raw_to_dfs(test_name)
    cleaned_transformed_dfs = [get_clean_df(df, mode=mode) for df in transformed_dfs]
    
    cleaned_dfs = []
    mutual_clean_rate = []
    mutual_real_amt = []
    mutual_amt = []
    
    for df, rate, real_amt, amt in cleaned_transformed_dfs:
        cleaned_dfs.append(df)
        mutual_clean_rate.append(rate)
        mutual_real_amt.append(real_amt)
        mutual_amt.append(amt)
    

    common_ids_set = common_ids(cleaned_dfs)
    mutual_dfs = [df[df["id"].isin(common_ids_set)] for df in cleaned_dfs]
    
    return mutual_dfs, mutual_clean_rate, mutual_real_amt, mutual_amt


def gen_subjective_quant_analysis(test_names: List[str], mode = "remove") -> dict:
    res = {}

    for test_name in test_names:
        
        mutual_dfs, mutual_clean_rate, mutual_real_amt, mutual_amt = get_subj_mutual_data(test_name=test_name, mode = mode)

        metric = ["accuracy", "logical", "clarity", "detail", "irrelevancy"]
        mean_scores = [df[metric].mean() for df in mutual_dfs]
        ovr_mean_scores = pd.concat(mean_scores, axis=1).mean(axis=1)
        ovr_std_scores = pd.concat(mean_scores, axis=1).std(axis=1)

        mean_scores = [mean_scores[i].to_dict() for i in range(len(mean_scores))]
        
        gwet_ac2 = gwet_AC2(test_name)

        res[test_name] = {
            "amount": mutual_amt[0], # TODO: improve
            "real_amount": mutual_real_amt[0], # TODO: improve
            "gen_rate": np.mean(mutual_clean_rate),
            "ovr_mean_scores": ovr_mean_scores.to_dict(),
            "ovr_std_scores": ovr_std_scores.to_dict(),
            "ovr_gwet_ac2": gwet_ac2["overall"],
            
            "mean_scores_per_sample": mean_scores,
            "gwet_per_metrics": gwet_ac2,
        }

    return res


def gen_quant_subj_df(
    test_names: List[str], export_detail: bool = False, mode = "replace"
) -> pd.DataFrame:
    
    quant_subj_res = gen_subjective_quant_analysis(test_names, mode = mode)
    res = {}
    
    for test_name, data in quant_subj_res.items():
        if export_detail:
            with open(f"./result/{test_name}/eval/quant_subj.json", "w") as f:
                json.dump(quant_subj_res[test_name], f)

        res[test_name] = {
            "amount": data["amount"],
            "real_amount": data["real_amount"],
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


def gen_subj_rank(test_names: List[str], mode = "replace") -> pd.DataFrame:
    quant_subj_res = gen_subjective_quant_analysis(test_names, mode = mode)
    res = {}
    for test_name, data in quant_subj_res.items():
        content = data["ovr_mean_scores"]
#         content["amount"] = data["amount"]
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


def merge_histogram(test_names):

    if not os.path.exists(MULTI_RESULT_PATH):
        os.makedirs(MULTI_RESULT_PATH)
        
    list_im = [f'./result/{sample_name}/eval/gen_size.jpg' for sample_name in test_names]
    
    imgs    = [ Image.open(i) for i in list_im ]
    imgs = [ Image.open("./result/human/histogram.jpg") ] + imgs
    min_shape = sorted( [(np.sum(i.size), i.size ) for i in imgs])[0][1]

    imgs_comb = np.vstack([i.resize(min_shape) for i in imgs])
    imgs_comb = Image.fromarray( imgs_comb)
    imgs_comb.save( f'./result/multieval/{TEST_NAME}/distribution.jpg' )
    
    return ""


def merge_prefix(test_names):

    if not os.path.exists(MULTI_RESULT_PATH):
        os.makedirs(MULTI_RESULT_PATH)

    list_im = [f'./result/{sample_name}/eval/prefix.jpg' for sample_name in test_names]
    
    imgs    = [ Image.open(i) for i in list_im ]
    imgs = [ Image.open("./result/human/prefix.jpg") ] + imgs
    min_shape = sorted( [(np.sum(i.size), i.size ) for i in imgs])[0][1]

    imgs_comb = np.hstack([i.resize(min_shape) for i in imgs])
    imgs_comb = Image.fromarray( imgs_comb)
    imgs_comb.save( f'./result/multieval/{TEST_NAME}/prefix.jpg' )
    
    return ""


def gen_quant_subj_chart(test_names: List[str], mode = "replace"):
    sns.set_style('whitegrid')
    
    # Adding title and labels
    if mode == "replace":
        title_suffix = "(Bad Data is Replaced with Minimum Score)"
    else:
        title_suffix = "(Bad Data is Removed)"
        
        
    
    ##### ---- Prepare datas ---- #####
    df = gen_quant_subj_df(test_names, mode = mode)[['avg_accuracy', 'avg_logical', 'avg_clarity','avg_detail', 'avg_irrelevancy']]
    
    # Flip irrelevancy to relevancy to match other metrics scale
    df['avg_relevancy'] = 4.0 - df['avg_irrelevancy']
    df.drop(columns = ['avg_irrelevancy'], inplace = True)
    
    
    # Concat with human's data
    df_human = pd.read_csv('./result/human/human_quant.csv', index_col=0)[['avg_accuracy', 'avg_logical', 'avg_clarity','avg_detail', 'avg_relevancy']]
    df = pd.concat([df, df_human], axis=1)
    
    # Create copy df columns (for Figure 1)
    df_overall = df.copy()
    df_overall['Score'] = df_overall.mean(axis=1)
    df_overall = df_overall.reset_index()
    df_overall.rename(columns={'index': 'Model'}, inplace=True)
    
    # Create copy melt df columns (for Figure 2)
    df_long = df.reset_index().melt(id_vars='index', value_vars=df.columns, var_name='Metric', value_name='Score')
    df_long.rename(columns={'index': 'Model'}, inplace=True)
    
    
    
    
    ###### ----- Plot 1st Figure ----- ######
    fig1 = plt.figure(figsize=(4,5))
    ax1 = sns.barplot(x="Model", y ="Score", data = df_overall, width = 0.3, color = '#85D1EC')
    
    plt.title('Final Score per Experiment ' + title_suffix, fontsize = 10,  pad=15)
    plt.xlabel('', fontsize = 10)
    plt.ylabel('Score', fontsize = 10)
    plt.ylim([2.0, 3.0])
    ax1.tick_params(axis='x', labelsize=8)
    ax1.tick_params(axis='y', labelsize=8)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)

    # Annotate each sub-bar with the score value
    for p in ax1.patches:
        ax1.annotate(f'{p.get_height():.2f}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', fontsize=8, color='black', xytext=(0, 5), 
                    textcoords='offset points')

    plt.tight_layout()
    
    # Save plot as bytes in memory
    fig1.savefig(f'./result/multieval/{TEST_NAME}/score_chart_{mode}.jpg', bbox_inches='tight')

    
    
    
    ###### ----- Plot 2nd Figure ----- ######
    fig2 = plt.figure(figsize=(10, 5))
    ax2 = sns.barplot(x='Model', y='Score', hue='Metric', data=df_long, width=0.75 )

    plt.title("Metric's Breakdown Score per Experiment " + title_suffix, fontsize = 10,  pad=15)
    plt.xlabel('', fontsize = 10)
    plt.ylabel('Score', fontsize = 10)
    plt.ylim([2.0, 3.0])
    ax2.tick_params(axis='x', labelsize=8)
    ax2.tick_params(axis='y', labelsize=8)
    ax2.legend(["accuracy", "logic", "clarity", "detail", "relevancy"])

#     # Rotate x-axis labels for better readability
#     plt.xticks(rotation=45)

    # Display the legend outside the plot
    plt.legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Annotate each sub-bar with the score value
    for p in ax2.patches:
        ax2.annotate(f'{p.get_height():.2f}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', fontsize=8, color='black', xytext=(0, 5), 
                    textcoords='offset points')

    plt.tight_layout()
    
    # Save plot as bytes in memory
    fig2.savefig(f'./result/multieval/{TEST_NAME}/quant_chart_{mode}.jpg', bbox_inches='tight')

    
    
    return ""


def gen_rate_chart(test_names, mode="replace"):
    df = gen_quant_subj_df(test_names, mode=mode)[['amount','real_amount','gen_rate']]
    df_human = pd.read_csv('./result/human/human_quant.csv', index_col=0)[['amount', 'real_amount', 'gen_rate']]
    
    
    df = pd.concat([df, df_human], axis=0)
    
    # Sample data
    categories = df.index.values
    bar_data = df.amount.values
    real_bar_data = df.real_amount.values
    line_data = df.gen_rate.values

    # Create figure and axes
    fig, ax1 = plt.subplots(figsize=(6, 6))

    # Plot bar chart for amount
    bar_width = 0.2
    bar_positions = np.arange(len(categories))
    bars1 = ax1.bar(bar_positions, bar_data, width=bar_width, label='Total Data Observed (Until 75 Valid Data)', color='#85D1EC')

    # Plot bar chart for real_amount
    bars2 = ax1.bar(bar_positions + bar_width, real_bar_data, width=bar_width, label='Valid Data Observed', color='#FFC300')

    # Add numbers on top of the bars
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval + 0.5, int(yval), ha='center', va='bottom', fontsize=10)
    
    for bar in bars2:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval + 0.5, int(yval), ha='center', va='bottom', fontsize=10)

    # Create secondary y-axis for line chart
    ax2 = ax1.twinx()
    ax2.plot(bar_positions + bar_width / 2, line_data, color='r', marker='o', label='Gen Rate')

    # Set labels and legend
    ax1.set_xlabel('Experiment', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax2.set_ylabel('Gen Rate', color='r', fontsize=12)
    ax1.legend(loc='upper left', fontsize = 8)
    ax2.legend(loc='upper right', fontsize = 10)

    # Rotate x-axis labels for ax1
    ax1.set_xticks(bar_positions + bar_width / 2)
    ax1.set_xticklabels(categories, rotation=45, ha='right', fontsize=10)

    # Set custom y-ticks and labels for ax1
    custom_yticks_ax1 = [50, 60, 70, 80, 90]  # Example y-tick values for ax1
    custom_ytick_labels_ax1 = ['50', '60', '70', '80', '90']  # Corresponding labels
    ax1.set_yticks(custom_yticks_ax1)
    ax1.set_yticklabels(custom_ytick_labels_ax1, fontsize=10)

    # Set custom y-ticks and labels for ax2
    custom_yticks_ax2 = [0.6, 0.7, 0.8, 0.9, 1.0]  # Example y-tick values for ax2
    custom_ytick_labels_ax2 = ['60%', '70%', '80%', '90%', '100%']  # Corresponding labels
    ax2.set_yticks(custom_yticks_ax2)
    ax2.set_yticklabels(custom_ytick_labels_ax2, fontsize=10)

    # Set y-limit
    ax1.set_ylim(50, max(bar_data) + 10)
    ax2.set_ylim(0.6, max(line_data) + 0.1)

    # Title
    plt.title('Amount and Generation Rate')
    plt.grid(False)

    # Save and show the plot
    plt.tight_layout()

    plt.savefig(f'./result/multieval/{TEST_NAME}/generation_rate_chart.jpg', bbox_inches='tight')
    
    return ""

