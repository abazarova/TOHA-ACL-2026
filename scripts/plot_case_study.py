import argparse
import json
import os
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from scipy.stats import entropy
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()

meta_dict = {
    "Mistral-7B-Instruct-v0.1": ("mistralai/Mistral-7B-Instruct-v0.1", 32),
    "Llama-2-7b-chat-hf": ("meta-llama/Llama-2-7b-chat-hf", 32),
    "Llama-2-13b-chat-hf": ("meta-llama/Llama-2-13b-chat-hf", 40),
    "Llama-3.1-8B-Instruct": ("meta-llama/Llama-3.1-8B-Instruct", 32),
    "Meta-Llama-3-8B-Instruct": ("meta-llama/Meta-Llama-3-8B-Instruct", 32),
    "Qwen2.5-7B-Instruct": ("Qwen/Qwen2.5-7B-Instruct", 28),
}


def load_mtd(row, i: int, j: int, root_dir: Path):
    sample_id = row["Unnamed: 0"]
    with open(root_dir / f"layer_{i}/head_{j}/{sample_id}.json") as f:
        data = json.load(f)
    return data["mtopdiv"] / data["response_len"]


def attn_map_to_dist_mx(attention_map, lower_bound=0.0):
    n_tokens = attention_map.shape[1]
    distance_mx = 1 - torch.clamp(
        attention_map.to(dtype=torch.float32), min=lower_bound
    )  # torch.where(attn_mx > lower_bound, attn_mx, 0.0)

    zero_diag = torch.ones(n_tokens, n_tokens) - torch.eye(n_tokens)
    distance_mx *= zero_diag
    distance_mx = torch.minimum(distance_mx.transpose(0, 1), distance_mx)
    return distance_mx.numpy()


def exact_token_entropy(attn_map, answ_pos):
    entropy_value = 0
    for i in range(answ_pos, attn_map.shape[0]):
        row = attn_map[i]
        entropy_value += entropy(row[:answ_pos] / row[:answ_pos].sum())
    return entropy_value / (attn_map.shape[0] - answ_pos + 1)


def distance_between_clusters(distance_mx, answ_pos):
    n_tokens = distance_mx.shape[-1]
    response_len = n_tokens - answ_pos
    adj_matrix = np.zeros((response_len + 1, response_len + 1))
    adj_matrix[0, 1:] = distance_mx[answ_pos:, :answ_pos].min(-1)
    adj_matrix[1:, 0] = distance_mx[answ_pos:, :answ_pos].min(-1)
    adj_matrix[1:, 1:] = distance_mx[answ_pos:, answ_pos:]

    graph = nx.from_numpy_array(adj_matrix)
    weights = [e[-1]["weight"] for e in graph.edges(0, data=True)]

    return min(weights)


def row_to_entropy(row, l: int, h: int):
    sample = row["entire_text"]
    input_ids = tokenizer(sample, return_tensors="pt")["input_ids"]
    answ_pos = (
        tokenizer(sample[: sample.rfind("A:")], return_tensors="pt")["input_ids"].shape[
            -1
        ]
        + 1
    )
    with torch.no_grad():
        outputs = llm(input_ids.cuda(), output_attentions=True)
        attention_map = outputs.attentions[l][0][h].cpu()

    return exact_token_entropy(attention_map, answ_pos)


def row_to_cluster_distance(row, l: int, h: int):
    sample = row["entire_text"]
    input_ids = tokenizer(sample, return_tensors="pt")["input_ids"]
    answ_pos = (
        tokenizer(sample[: sample.rfind("A:")], return_tensors="pt")["input_ids"].shape[
            -1
        ]
        + 1
    )
    with torch.no_grad():
        outputs = llm(input_ids.cuda(), output_attentions=True)
        attention_map = outputs.attentions[l][0][h].cpu()
        distance_mx = attn_map_to_dist_mx(attention_map=attention_map)

    return distance_between_clusters(distance_mx, answ_pos)


def plot():
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Plot heatmaps for the analysis of MTop-Div behaviour",
        description="Calculate and plot attention map features",
    )

    parser.add_argument("model_name")
    parser.add_argument("dataset_name")
    parser.add_argument(
        "--entropy",
        "-e",
        action="store_true",
        help="Calculate entropy of response-prompt attention",
    )
    parser.add_argument(
        "--cluster-distance",
        "-c",
        action="store_true",
        help="Calculate prompt-response cluster distance",
    )
    parser.add_argument("--save-path", default="../plot/")
    parser.add_argument("--mtd-path", default="/app/cache/mtopdiv")
    parser.add_argument("--device", default="cuda")

    args = parser.parse_args()

    login(os.getenv("HUGGING_FACE_API_KEY"))

    model_name = args.model_name
    model_id, n_l = meta_dict[model_name]
    llm = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)
    device = args.device
    llm = llm.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    ds_name = args.dataset_name
    df = pd.read_csv(f"../data/raw/{ds_name}/{ds_name.lower()}_{model_name}.csv")
    if ds_name == "CoQA":
        df["entire_text"] = df.apply(
            lambda sample: sample["context"]
            + " Q: "
            + sample["question"]
            + " A: "
            + sample["generated_answer"],
            axis=1,
        )
    else:
        raise NotImplementedError
    save_path = Path(args.save_path)
    mtd_path = Path(args.mtd_path) / (ds_name.lower() + "/zero_out_prompt/" + model_name)

    for i, row in tqdm(df.iterrows()):
        sample = row["entire_text"]
        input_ids = tokenizer(sample, return_tensors="pt")["input_ids"]
        answ_pos = (
            tokenizer(sample[: sample.rfind("A:")], return_tensors="pt")[
                "input_ids"
            ].shape[-1]
            + 1
        )
        with torch.no_grad():
            outputs = llm(input_ids.cuda(), output_attentions=True)
            attention_maps = outputs.attentions
        for l, h in zip(np.arange(n_l), np.arange(n_l)):
            attn_map = attention_maps[l][0][h].cpu()
            df.at[i, f"mtd_{l}_{h}"] = load_mtd(row, l, h, mtd_path)
            if args.entropy:
                df.at[i, f"entropy_{l}_{h}"] = exact_token_entropy(attn_map, answ_pos)
            if args.cluster_distance:
                distance_mx = attn_map_to_dist_mx(attention_map=attn_map)
                df.at[i, f"cluster_distance_{l}_{h}"] = distance_between_clusters(
                    distance_mx, answ_pos
                )

    df.to_csv(save_path / f"{ds_name.lower()}_{args.model_name}.csv")
