import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import mtd.barcodes as mtd
import numpy as np
import ripserplusplus as rpp_py
import scipy.stats as st
import torch
from dotenv import load_dotenv
from gudhi.wasserstein import wasserstein_distance
from huggingface_hub import login
from hydra import compose, initialize
from hydra.utils import instantiate
from joblib import Parallel, delayed
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda"

load_dotenv()
names_dict = {
    "Mistral-7B-Instruct-v0.1": "mistralai/Mistral-7B-Instruct-v0.1",
    "Llama-2-7b-chat-hf": "meta-llama/Llama-2-7b-chat-hf",
    "Llama-2-13b-chat-hf": "meta-llama/Llama-2-13b-chat-hf",
    "Llama-3.1-8B-Instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "Meta-Llama-3-8B-Instruct": "meta-llama/Meta-Llama-3-8B-Instruct",
    "Qwen2.5-7B-Instruct": "Qwen/Qwen2.5-7B-Instruct",
}


def transform_attention_scores_to_distances(
    attn_mxs: torch.Tensor,
    lower_bound: float = 0.0,
) -> torch.Tensor:
    """Transform attention matrix to the matrix of distances between tokens.

    Parameters
    ----------
    attn_mxs : torch.Tensor
        Attention matrixes of one sample (n_heads x n_tokens x n_tokens).
    lower_bound: float
        Zero out values smaller than this value.

    Returns
    -------
    torch.Tensor
        Distance matrix.

    """
    n_tokens = attn_mxs.shape[1]
    distance_mx = 1 - torch.clamp(
        attn_mxs, min=lower_bound
    )  # torch.where(attn_mx > lower_bound, attn_mx, 0.0)
    zero_diag = torch.ones(n_tokens, n_tokens) - torch.eye(n_tokens)
    distance_mx *= zero_diag.to(attn_mxs.device).expand_as(
        distance_mx
    )  # torch.diag(torch.diag(distance_mx))
    distance_mx = torch.minimum(distance_mx.transpose(1, 2), distance_mx)
    return distance_mx.cpu().numpy()


def bos_attention(matrix: np.ndarray, response_len: int) -> float:
    """Calculate ratio of attention to <bos> token (first token in the sequence).

    Args:
        matrix (np.ndarray): a square matrix of shape (n_tokens, n_tokens)
        response_len (int): length of the response

    Returns:
        float: Attention to <bos> ratio.

    """
    return matrix[-response_len:, 0].sum() / response_len


def sparsity_ratio(matrix: np.ndarray, threshold: float = 0.05) -> float:
    (
        """Calculate the ratio of values smaller than a predefined threshold in the matrix.

    Args:
        matrix (np.ndarray): a square matrix of shape (n_tokens, n_tokens)
        threshold (float, optional): a predefined threshold. Defaults to 0.05.

    Returns:
        float
            Sparsity ratio.

    """
        """"""
    )
    n_tokens = matrix.shape[0]
    total = n_tokens * (n_tokens + 1) / 2

    return (matrix <= threshold).sum() / total


def entropy(attn_map: np.ndarray, average: str = "max") -> float:
    """Calculate entropies of each row in the attention map and aggregate them according to the "average" parameter value.

    Args:
        attn_map (np.ndarray): attention map, square matrix of shape (n_tokens, n_tokens)
        average (str, optional): aggregation regime ("min", "max" or "mean"). Defaults to "max".

    Raises:
        NotImplementedError: if average is not equal to "min", "max" or "mean".

    Returns:
        _type_: float

    """
    entropies = st.entropy(attn_map, axis=1)

    if average == "max":
        return np.max(entropies)
    elif average == "min":
        return np.min(entropies)
    elif average == "mean":
        return np.mean(entropies)
    else:
        raise NotImplementedError


def attention_distance(attn_map: np.ndarray) -> float:
    """Calculate attention_distance for the attention map. See https://aclanthology.org/W19-4808.pdf.

    Args:
        attn_map (np.ndarray): attention map, square matrix of shape (n_tokens, n_tokens)

    Returns:
        float: attention_distance value

    """
    pass


def wasserstein_bw_diag(attn_map: np.ndarray, prompt_len: int) -> float:
    """Calculate Wasserstein distance bw persistence diagrams of prompt graph and the entire attention graph.

    Args:
        attn_map (np.ndarray): attention map
        prompt_len (int): length of the prompt

    Returns:
        float: wasserstein distance

    """
    barcodes = rpp_py.run("--format distance --dim 1", attn_map)
    barcodes = np.array(mtd.barc2array(barcodes)[0], dtype=float)

    barcodes_prompt = rpp_py.run(
        "--format distance --dim 1", attn_map[:prompt_len, :prompt_len]
    )
    barcodes_prompt = np.array(mtd.barc2array(barcodes_prompt)[0], dtype=float)

    if barcodes_prompt.size and barcodes.size:
        return wasserstein_distance(barcodes, barcodes_prompt)
    else:
        return wasserstein_distance(barcodes, np.array([[0, 0]]))


def spectral_norm(attn_map: np.ndarray):
    """Calculate spectral norm of the attention map.

    Args:
        attn_map (np.ndarray): square matrix (n_tokens, n_tokens).

    Returns:
        float: spectral norm value.

    """
    return np.linalg.norm(attn_map.astype(np.float32), ord=2)


def calc_features(
    attn_map: np.ndarray,
    triu: np.ndarray,
    threshold: float = 0.05,
    average: str = "max",
):
    """Calculate features for the attention map.

    Args:
        attn_map (np.ndarray): attention map (square matrix of shape n_tokens x n_tokens).
        triu (np.ndarray): upper-triangular matrix of the same size as attn_map.
        threshold (float, optional): sparsity ratio threshold. Defaults to 0.05.
        average (str, optional): aggregtion model for entropy. Defaults to "max".

    Returns:
        (sr, ent, sp_norm): feature values (sparsity ratio, row-wise entropy, spectral norm)

    """
    sr = sparsity_ratio(attn_map + triu, threshold=threshold)
    ent = entropy(attn_map, average=average)
    sp_norm = np.linalg.norm(attn_map.astype(np.float32), ord=2)

    return sr, ent, sp_norm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str, default="Mistral-7B-Instruct-v0.1")
    parser.add_argument("preprocess", type=str, default="ragtruth_qa")
    parser.add_argument(
        "--save_dir", type=str, default="/app/cache/traditional_features"
    )
    parser.add_argument(
        "--wasserstein", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--sp_norm", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--entropy", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--variability", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--attn_distance", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--sparsity_ratio", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--bos_attention", action=argparse.BooleanOptionalAction, default=False
    )

    args = parser.parse_args()
    print(args)

    dataset_name = args.preprocess
    model_name = args.model_name
    save_dir = Path(args.save_dir) / f"bos_features_{dataset_name}_{model_name}.npy"

    with initialize(version_base=None, config_path="./config"):
        cfg = compose(
            config_name="master",
            overrides=[
                f"preprocess={dataset_name}",
                f"model_name={model_name}",
            ],
        )

    dataset = instantiate(cfg["preprocess"])
    X, y, train_indices, test_indices = dataset.process()

    model_id = names_dict[model_name]
    login(os.getenv("HUGGING_FACE_API_KEY"))

    llm = AutoModelForCausalLM.from_pretrained(model_id)
    llm = llm.half().to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    features = defaultdict(list)
    n_layers = 32
    n_heads = 32
    n_jobs = 8
    magic_value = 10  # see https://aclanthology.org/W19-4808.pdf.

    avg_attn_maps = np.zeros((n_layers * n_heads, magic_value, magic_value))
    avg_attn_dist = np.zeros((n_layers * n_heads,))
    variability = np.zeros(
        (n_layers * n_heads,)
    )  # see https://aclanthology.org/W19-4808.pdf
    normalizing_const = 0
    for _, sample in tqdm(X.iterrows(), total=len(X)):
        prompt = sample["prompt"]
        answer = sample["response"]

        prompt_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
        answer_ids = tokenizer(answer, add_special_tokens=False, return_tensors="pt")
        input_ids = torch.cat(
            [prompt_ids["input_ids"], answer_ids["input_ids"]], axis=1
        ).to(device)

        # Yield the output of the model for the current example
        with torch.no_grad():
            output = llm(
                input_ids,
                output_hidden_states=False,
                output_attentions=True,
            )

        n_tokens = len(input_ids[0])
        triu = np.triu(np.ones((n_tokens, n_tokens))) - np.eye(n_tokens)
        attn_maps = [
            elem[0][i].cpu() for i in range(32) for elem in output.attentions
        ]  # n_layers x n_heads
        attn_maps = np.array(attn_maps)
        if args.variability:
            avg_attn_maps += attn_maps[:, :magic_value, :magic_value]

        if args.attn_distance:
            normalizing_const += n_tokens

        if args.wasserstein:
            distance_mxs = transform_attention_scores_to_distances(
                torch.stack(attn_maps)
            )
            wasserstein_tmp = Parallel(n_jobs=n_jobs)(
                delayed(wasserstein_bw_diag)(attn_map, len(prompt_ids))
                for attn_map in distance_mxs
            )
            features["wasserstein"].append(wasserstein_tmp)
        if args.sp_norm:
            sp_norm_tmp = Parallel(n_jobs=n_jobs)(
                delayed(spectral_norm)(attn_map) for attn_map in attn_maps
            )
            features["spectral_norm"].append(sp_norm_tmp)
        if args.entropy:
            entropy_tmp = Parallel(n_jobs=n_jobs)(
                delayed(entropy)(attn_map, "mean") for attn_map in attn_maps
            )
            features["entropy"].append(entropy_tmp)
        if args.sparsity_ratio:
            sparsity_tmp = Parallel(n_jobs=n_jobs)(
                delayed(sparsity_ratio)(attn_map, 0.05) for attn_map in attn_maps
            )
            features["sparsity_ratio"].append(sparsity_tmp)

        if args.bos_attention:
            bos_attention_tmp = Parallel(n_jobs=n_jobs)(
                delayed(bos_attention)(attn_map, len(answer_ids[0])) for attn_map in attn_maps
            )
            features["bos_attention"].append(bos_attention_tmp)

    np.save(save_dir, features)

    if args.variability or args.attn_distance:
        for _, samples in tqdm(X.iterrows(), total=len(X)):
            prompt = sample["prompt"]
            answer = sample["response"]

            prompt_ids = tokenizer(
                prompt, add_special_tokens=False, return_tensors="pt"
            )
            answer_ids = tokenizer(
                answer, add_special_tokens=False, return_tensors="pt"
            )
            input_ids = torch.cat(
                [prompt_ids["input_ids"], answer_ids["input_ids"]], axis=1
            ).to(device)

            with torch.no_grad():
                output = llm(
                    input_ids,
                    output_hidden_states=False,
                    output_attentions=True,
                )

            n_tokens = len(input_ids[0])
            attn_maps = [
                elem[0][i].cpu() for i in range(32) for elem in output.attentions
            ]  # (n_layers x n_heads, n_tokens, n_tokens)

            if args.attn_distance:
                dist_map = np.zeros((n_tokens, n_tokens))
                for i in range(n_tokens):
                    dist_map[i, : i + 1] = np.arange(i + 1)[::-1]
                avg_attn_dist += (
                    np.matmul(np.array(attn_maps), dist_map).sum((-1, -2))
                    / normalizing_const
                )
            if args.variability:
                variability += np.abs(
                    np.array(attn_maps)[:, :magic_value, :magic_value]
                    - avg_attn_maps / len(X)
                ).sum() / (2 * magic_value * len(X))

        if args.variability:
            features["variability"] = variability
        if args.attn_distance:
            features["attn_distance"] = avg_attn_dist

    #np.save(save_dir, features)
