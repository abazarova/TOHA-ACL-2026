import os
from typing import List, Any
import hashlib
from functools import wraps
import warnings

import numpy as np
import pandas as pd
from scipy.stats import rankdata
import torch
from tqdm import trange
from loguru import logger
from joblib import dump, load



def gershgorin_disks(M):
    diag = torch.diag(M).real
    centers = diag.cpu().numpy()
    radii = torch.sum(torch.abs(M), dim=1).cpu().numpy() - np.abs(centers)
    return centers, radii

def iqr_outlier_count(points):
    if len(points) == 0:
        return 0
    q1 = np.percentile(points, 25)
    q3 = np.percentile(points, 75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return int(np.sum((points < lower) | (points > upper)))

@torch.no_grad()
def copying_head_scores(
    model,
    num_chunks: int = 8,
    chunk_size: int | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
):
    """
    Compute per-head scores (trace + Gershgorin outlier counts) in a memory-friendly way.

    - trace computed by cyclic property on HxH matrices
    - Gershgorin disks computed by chunking rows of M = Wu @ Wov @ We.T
    """

    num_layers = len(model.model.layers)
    hidden_size = model.config.hidden_size
    num_heads = model.config.num_attention_heads
    head_dim = hidden_size // num_heads
    num_value_heads = model.config.num_key_value_heads

    # Base weights
    Wu = model.lm_head.weight       # [V, H]
    We = model.model.embed_tokens.weight  # [V, H]

    # device/dtype selection
    if device is None:
        device = Wu.device
    if dtype is None:
        dtype = Wu.dtype

    # Move (or ensure) tensors on chosen device & dtype
    Wu = Wu.to(device=device, dtype=dtype)
    We = We.to(device=device, dtype=dtype)

    V, H = Wu.shape
    if chunk_size is None:
        chunk_size = max(1, V // num_chunks)

    traces, abs_traces, outlier_counts = [], [], []

    for layer_idx in trange(num_layers, desc="Iterating layers", total=num_layers):
        layer = model.model.layers[layer_idx]
        Wo = layer.self_attn.o_proj.weight.to(device=device, dtype=dtype)   # [H, H]
        Wv = layer.self_attn.v_proj.weight.to(device=device, dtype=dtype)   # [H, H]

        # expand key/value heads if model uses kv-sharing
        Wv = Wv.repeat_interleave(num_heads // num_value_heads, dim=0)

        for head_idx in trange(num_heads, desc=f"Layer {layer_idx}: Iterating heads", total=num_heads, leave=False):
            # Wov: [H, H]
            s = head_idx * head_dim
            e = (head_idx + 1) * head_dim
            Wov = Wv[:, s:e].matmul(Wo[s:e, :])

            # --- Gershgorin via chunking (we must still look at rows of full M)
            centers_list = []
            row_abs_sums_list = []

            for start in range(0, V, chunk_size):
                end = min(V, start + chunk_size)
                Wu_chunk = Wu[start:end]                 # [chunk, H]
                # (Wu_chunk @ Wov) -> [chunk, H], then @ We.T -> [chunk, V]
                M_chunk = Wu_chunk.matmul(Wov).matmul(We.t())  # [chunk, V]

                rows = end - start
                # vectorized diagonal pick: local_row_idx -> global_col_idx = start + local_row_idx
                local_rows = torch.arange(rows, device=device)
                global_cols = torch.arange(start, end, device=device)
                diag_chunk = M_chunk[local_rows, global_cols]           # [rows]
                row_abs_sums = torch.sum(torch.abs(M_chunk), dim=1)     # [rows]

                centers_list.append(diag_chunk.cpu().numpy())
                row_abs_sums_list.append(row_abs_sums.cpu().numpy())

                # free M_chunk early (optional)
                del M_chunk

            centers = np.concatenate(centers_list)
            tr_val = sum(centers)
            traces.append(tr_val)
            abs_traces.append(abs(tr_val))

            row_abs_sums = np.concatenate(row_abs_sums_list)
            radii = row_abs_sums - np.abs(centers)
            boundary_points = np.concatenate([centers + radii, centers - radii])
            outlier_counts.append(iqr_outlier_count(boundary_points))

    traces = np.array(traces)
    abs_traces = np.array(abs_traces)
    outlier_counts = np.array(outlier_counts)

    rank_outliers = rankdata(outlier_counts, method="average")   # few outliers → good
    rank_trace = rankdata(-traces, method="average")            # large trace → good
    scores = rank_outliers + rank_trace

    return scores

def get_dataframe_hash(X: pd.DataFrame) -> str:
    """Compute a hash for the DataFrame to check if cached hiddens match.

    Parameters
    ----------
    X : pd.DataFrame
        Input DataFrame to compute the hash.

    Returns
    -------
    str : Hash value representing the contents of the DataFrame.

    """
    X_bytes = X.to_string().encode()
    return hashlib.md5(X_bytes).hexdigest()

def reshape(x: List[List[List[Any]]], batch_first=True):
    if batch_first:
        return [
        [
            [x[b][i][h] for i in range(len(x[b]))]
            for b in range(len(x))
        ]
        for h in range(len(x[0][0]))
    ]
    return [
        np.array([
            [x[h][b][i] for h in range(len(x))]
            for i in range(len(x[0][b]))
        ]) for b in range(len(x[0]))
    ]
    
def cache_redeep(cache_dir: str, message: str = "Processing"):
    """Cache the result of a function call.

    Note that you need to pass argument "hash" or "cache_name" as keyword argument whenever you call wrapped function.

    Args:
    ----
        default_cache_dir: The directory where the cache will be stored.
        If cache_dir keyword argument is passed in wrapped function call,
        then it will supersede value specified in decorator
        message: A message to log when processing.

    Returns:
    -------
        The wrapped function with caching enabled.

    """

    def decorator(func):
        assert func.__name__ == "compute_redeep_scores", print(f'This caching wrapper does not accept {func.__name__}')
        @wraps(func)
        def wrapper(*args, **kwargs):
            hash = kwargs.pop("hash", None)
            cache_name = kwargs.pop("cache_name", None)
            cache_dir_from_func_call = kwargs.pop("cache_dir", None)
            used_cache_dir = (
                cache_dir_from_func_call
                if cache_dir_from_func_call is not None
                else cache_dir
            )
            if (hash is None) and (cache_name is None):
                raise ValueError(
                    "You need to pass hash as keyword argument or pass cachefile name"
                )

            if cache_name is None:
                cache_file = os.path.join(used_cache_dir, f"{func.__name__}_{hash}.pkl")
            else:
                cache_file = os.path.join(used_cache_dir, cache_name)

            os.makedirs(used_cache_dir, exist_ok=True)

            target_heads = tuple(map(tuple, kwargs["copy_heads"]))
            target_layers = tuple(kwargs["knowledge_ffns"])

            external_similarity = dict()
            knowledge_diff = dict()

            if os.path.exists(cache_file):
                logger.info(f"Loading cached result from {cache_file}.")
                # breakpoint()
                with open(cache_file, "rb") as f:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", FutureWarning)
                        data = load(f)
                logger.info(f"Successfully loaded cached result from {cache_file}")
                logger.info(f"Identifying missing values") 

                missing_heads, missing_layers = [], []
                for key in target_heads:
                    external_similarity[key] = data['external_similarity'].get(key, [])
                    if len(external_similarity[key]) == 0:
                        missing_heads.append(key)
                for key in target_layers:
                    knowledge_diff[key] = data['knowledge_diff'].get(key, [])
                    if len(knowledge_diff[key]) == 0:
                        missing_layers.append(key)
            else:
                logger.info(
                    f"{message}: No cache found, for {cache_file} running {func.__name__}."
                )
                data = dict()
                data['external_similarity'] = dict()
                data['knowledge_diff'] = dict()
                missing_heads = target_heads
                missing_layers = target_layers
                    
            if len(missing_heads) == 0 and len(missing_layers) == 0:
                logger.info(f"There is no missing values")
            else:
                logger.info(f"Filling in missing values")
                kwargs["copy_heads"] = missing_heads
                kwargs["knowledge_ffns"] = missing_layers
                result = func(*args, **kwargs)

                missing_external_similarity, missing_knowledge_diff = result
                if len(missing_heads) > 0:
                    missing_external_similarity = reshape(missing_external_similarity, batch_first=True)
                if len(missing_layers) > 0:
                    missing_knowledge_diff = reshape(missing_knowledge_diff, batch_first=True)
            
                for i, key in enumerate(missing_heads):
                    external_similarity[key] = missing_external_similarity[i]
                for i, key in enumerate(missing_layers):
                    knowledge_diff[key] = missing_knowledge_diff[i]

                logger.info(f"Saving result to {cache_file}")

                data['external_similarity'].update(external_similarity)
                data['knowledge_diff'].update(knowledge_diff)

                with open(cache_file, "wb") as f:
                    dump(data, f)
                logger.info(f"{message}: Saved result to {cache_file}.")

            return (
                reshape(list(external_similarity.values()), batch_first=False),
                reshape(list(knowledge_diff.values()), batch_first=False),
            )

        return wrapper

    return decorator

