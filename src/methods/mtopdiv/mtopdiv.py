from collections import defaultdict
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Literal, Union, Optional

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from ..hallucination_detection_abc import HallucinationDetectionMethod
from ..llm_base import LLMBase
from .utils import (
    select_best_features,
    get_mtopdivs,
)
from ..performance_monitor import log_execution_time
from ..caching_utils import cache_result, get_dataframe_hash


# logger.disable("src.methods.caching_utils")

@cache_result(cache_dir="cache/mtopdiv", message="Get MTopDiv scores")
def compute_scores(
    X: pd.DataFrame,
    model: LLMBase,
    analysis_sites,
    n_jobs,
    backend="processes",
    hash: Optional[str] = None,
    cache_name: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> list[list[float]]:
    attns_ids_dict = dict()
    layers = sorted(np.unique([x[0] for x in analysis_sites]).tolist())
    for layer in layers:
        attns_ids_dict[layer] = sorted([x[1] for x in analysis_sites if x[0] == layer])
    mtopdiv_list = []
    for output, _, answer_ids in tqdm(model.generate_llm_outputs(
        X, output_hidden_states=False, output_attentions=True
    ), total=len(X), desc='Iterating over samples', leave=False):
        attn_heads = tuple([
            output["attentions"][layer_idx][0, heads_idxs].cpu().numpy() for layer_idx, heads_idxs in attns_ids_dict.items()
        ])

        response_length = len(answer_ids[0])
        mtopdivs = get_mtopdivs(
            attn_heads,
            response_length,
            n_jobs,
            backend,
        )
        mtopdiv_list.append(mtopdivs)
    return mtopdiv_list


@dataclass
class MTopDiv(HallucinationDetectionMethod):
    """A class to compute MTopDivergence between tokens of a prompt and a response.

    This class provides methods to calculate MTopDivergence (MTopDiv) for given text prompts and responses using
    pre-trained language models.

    Attributes
    ----------
    model_name : Literal["Llama-2-7b-chat-hf", "Mistral-7B-Instruct-v0.1"]
        The name of the pre-trained model to use for computing divergences. Choices are 'Llama-2-7b-chat-hf' or
        'Mistral-7B-Instruct-v0.1'.
    dtype : str
        The data type used for the LLM inference, e.g., 'float16', 'float32'. Determines the precision of computations.
    device : str
        The device to run the model on, such as 'cuda' for GPU or 'cpu' for CPU. Default is 'cuda'.
    cache_dir : str
        Directory to save or load precomputed MTopDiv data. If this directory does not exist, it will be created.
        Default is 'cache/mtopdiv'.

    mode : Literal["supervised", "unsupervised"]
        Specifies whether to evaluate the obtained results in a supervised or unsupervised mode.
    analysis_sites : list[tuple[int, int]]
        List of tuples representing pairs of (layer_index, head_index) that are of interest.
    zero_out : Literal["prompt", "response"]
        Determines whether to zero out distances between prompt tokens or response tokens.
    normalize_by_length : bool
        Indicates whether to divide the obtained MTopDiv values by the length of the response or prompt
        (depending on which is zeroed out). Default is `False`.

    """

    model_name: Literal["Llama-2-7b-chat-hf", "Mistral-7B-Instruct-v0.1"]
    dtype: str = "float16"
    device: str = "cuda"
    cache_dir: str = "cache/mtopdiv/ragtruth_qa"

    mode: Literal["supervised", "unsupervised"] = "supervised"
    analysis_sites: Union[Literal["all"], list[tuple[int, int]]] = "all"

    n_jobs: int = 16
    backend: Literal["threads", "processes"] = "threads"
    n_layers: int = 32
    n_heads: int = 32
    n_max: int = 6  # hyperparameter

    def __post_init__(self):
        """Post initialization of the class."""
        if self.analysis_sites != "all":
            self.analysis_sites = list(map(list, self.analysis_sites))
            self.analysis_sites = sorted(self.analysis_sites)
        self.clf = None
        self.llm_model = LLMBase(self.model_name, self.dtype, self.device)

    @log_execution_time(use_cuda_timing=True)
    def transform(self, X: pd.DataFrame) -> list[list[float]]:
        """Calculate MTopDiv for each entry in the DataFrame.

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame containing 'prompt' and 'response' columns.

        Returns
        -------
        list
            List of MTopDiv values for each entry.

        """
        data_hash = get_dataframe_hash(X)
        heads_hash = hash(tuple(map(tuple, self.analysis_sites)))
        cache_name = f"{self.model_name}_{data_hash}_{heads_hash}_mtopdiv.joblib"

        if self.llm_model.llm is None:
            self.llm_model.llm, self.llm_model.tokenizer = self.llm_model.instantiate_llm()
        
        mtopdivs = compute_scores(
            X=X,
            model=self.llm_model,
            analysis_sites=self.analysis_sites,
            n_jobs=self.n_jobs,
            backend=self.backend,
            cache_name=cache_name,
            cache_dir=self.cache_dir,
        )

        return mtopdivs

    def fit(
        self,
        X_train: list[list[float]],
        y_train: list[int],
        *args
    ) -> "MTopDiv":
        """Train the logistic regression on the MTopDiv features for provided dataset.

        Parameters
        ----------
        X_train: list
            List containing the training data.
        y_train : list
            List containing the target labels.

        Returns
        -------
        self : MTopDiv
            Returns the instance of the class with the trained model.

        """
        return self

    def predict_score(self, X: list[list[float]]) -> np.ndarray[float]:
        """Perform inference.

        Parameters
        ----------
        X : list[list[float]]
            List of lists of MTopDiv values from fixed model heads for each sample in the dataset.

        Returns
        -------
        List[float]
            List of predicted probabilities for each input example.

        """
        X = np.array(X)

        if self.mode == "supervised":
            logger.info("MTopDiv-based classifier prediction is used as a hallucination score.")
            return self.clf.predict_proba(X)[:, 1]
        elif self.mode == "unsupervised":
            logger.info("Average MTopDiv is used as a hallucination score.")
            return np.abs(X).mean(axis=-1)
        else:
            raise ValueError("This mode is not supported. Please selected either 'supervised' or 'unsupervised' mode.")

    def fit_hyperparameters(self, X_val: pd.DataFrame, y_val: pd.DataFrame):
        """Select optimal head subset given the probe dataset."""
        self.analysis_sites = product(range(self.n_layers), range(self.n_heads))
        self.analysis_sites = sorted(self.analysis_sites)
        
        features = self.transform(X_val)
        features = np.array(features)  # (n_samples, n_layers * n_heads)

        columns = [f"{i}_{j}" for i, j in self.analysis_sites]
        df = pd.DataFrame(features, columns=columns)

        if self.mode == "supervised":
            best_auc = 0
            for n in range(1, self.n_max + 1):
                # Use Recursive Feature Elimination to select top n features
                selected_features = select_best_features(df, y_val, n)
                # Train model on selected features
                clf_temp = LogisticRegression()
                clf_temp.fit(df[selected_features], y_val)

                # Predict on the same data and calculate ROC AUC
                y_pred_proba = clf_temp.predict_proba(df[selected_features])[:, 1]
                auc_score = roc_auc_score(y_val, y_pred_proba)

                # Check if this is the best performance so far
                if auc_score > best_auc:
                    best_auc = auc_score
                    best_features = selected_features
                    best_clf = clf_temp

            self.clf = best_clf
            f = lambda elem: (int(elem[0]), int(elem[1]))
            self.analysis_sites = [f(elem.split("_")) for elem in best_features]
        elif self.mode == "unsupervised":
            df["is_hal"] = y_val.values

            avg_distances = pd.DataFrame(
                columns=np.arange(self.n_heads), index=np.arange(self.n_layers)
            )
            hallu_mtd = df.loc[:, columns][df["is_hal"] == 1].apply(np.mean, axis=0)
            grnd_mtd = df.loc[:, columns][df["is_hal"] == 0].apply(np.mean, axis=0)
            for l, h in self.analysis_sites:
                avg_distances.at[l, h] = hallu_mtd[f"{l}_{h}"] - grnd_mtd[f"{l}_{h}"]
            dist_copy = avg_distances.copy()

            optimal_subset = []
            best_auroc, n_opt = 0, 0
            for n in range(1, self.n_max + 1):
                best_pos = np.unravel_index(np.argmax(dist_copy), dist_copy.shape)  # (h, l)
                optimal_subset.append(best_pos)
                dist_copy[best_pos[1]][best_pos[0]] = -1
                predictions = df[[f"{layer}_{head}" for layer, head in optimal_subset]].mean(axis=1)
                roc_auc = roc_auc_score(y_val, predictions)
                if roc_auc > best_auroc:
                    n_opt = n
                    best_auroc = roc_auc
            self.analysis_sites = optimal_subset[:n_opt]
        else:
            raise ValueError("This mode is not supported. Please selected either 'supervised' or 'unsupervised' mode.")

        print("SELECTED HEADS:", self.analysis_sites)