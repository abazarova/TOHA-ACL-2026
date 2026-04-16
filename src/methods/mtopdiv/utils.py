import os
from typing import List, Optional
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from ripser import ripser




os.environ["TOKENIZERS_PARALLELISM"] = "false"


def select_best_features(X: pd.DataFrame, y: pd.Series, 
                        n_features: int = 10, 
                        method: str = 'f_classif') -> list:
    """
    Select best features using univariate statistical tests
    
    Args:
        X: DataFrame with features (columns are feature names)
        y: Series with true labels
        n_features: Number of top features to select
        method: 'f_classif' (ANOVA F-value) or 'mutual_info_classif' (mutual information)
    
    Returns:
        List of selected feature names
    """
    # Validate inputs
    if not isinstance(X, pd.DataFrame):
        raise ValueError("X must be a pandas DataFrame")
    if not isinstance(y, pd.Series):
        raise ValueError("y must be a pandas Series")
    if len(X) != len(y):
        raise ValueError("X and y must have the same number of samples")

    # Choose scoring method
    if method == 'f_classif':
        scorer = f_classif
    elif method == 'mutual_info_classif':
        scorer = mutual_info_classif
    else:
        raise ValueError("method must be 'f_classif' or 'mutual_info_classif'")

    # Select top features
    selector = SelectKBest(score_func=scorer, k=min(n_features, X.shape[1]))
    selector.fit(X, y)

    # Get selected feature names
    selected_mask = selector.get_support()
    selected_features = X.columns[selected_mask].tolist()

    return selected_features


def transform_attention_scores_to_distances(
    attention_weights: np.ndarray,
) -> np.ndarray:
    """Transform attention matrix to the matrix of distances between tokens.

    Parameters
    ----------
    attention_weights : np.ndarray
        Attention matrices of one sample (n_heads x n_tokens x n_tokens).

    Returns
    -------
    np.ndarray
        Distance matrix.

    """
    attention_weights = attention_weights.astype(np.float32)
    n_tokens = attention_weights.shape[-1]
    distance_mx = 1 - np.clip(attention_weights, a_min=0.0, a_max=None)
    zero_diag = np.ones((n_tokens, n_tokens)) - np.eye(n_tokens)

    distance_mx *= np.broadcast_to(zero_diag, distance_mx.shape)
    distance_mx = np.minimum(
        np.swapaxes(distance_mx, -1, -2),
        distance_mx,
    )

    return distance_mx


def transform_distances_to_mtopdiv(distance_mx: np.ndarray) -> float:
    """
    Compute the MTopDiv (Manifold Topology Divergence) score from a distance matrix.

    This function calculates the sum of persistence intervals in the H₀ (zero-dimensional)
    persistent homology barcode, corresponding to the lifetimes of connected components
    in a Vietoris–Rips filtration.

    Parameters:
        distance_mx (np.ndarray): A square, symmetric distance matrix.

    Returns:
        float: Sum of finite H₀ barcode lengths (birth–death), representing topological diversity.
    """
    barcodes = ripser(distance_mx, distance_matrix=True, maxdim=0)["dgms"]
    if len(barcodes) > 0:
        return barcodes[0][:-1, 1].sum()
    return 0

def get_mtopdivs(
    attns: List[np.array],
    response_length: int,
    n_jobs: Optional[float] = 1,
    backend="processes",
) -> np.ndarray:
    def job(layer_head_pair):
        layer, head = layer_head_pair
        distance_matrix = transform_attention_scores_to_distances(
            attns[layer][head]
        )
        distance_matrix[:-response_length, :-response_length] = 0
        mtopdiv = transform_distances_to_mtopdiv(distance_matrix) / response_length
        return mtopdiv
    attn_heads_idx = [(layer, head) for layer in range(len(attns)) for head in range(attns[layer].shape[0])]
    with Parallel(n_jobs=n_jobs, prefer=backend) as parallel:
        mtopdivs = parallel(delayed(job)((layer, head)) for layer, head in attn_heads_idx)
    return mtopdivs
