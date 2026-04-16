from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

import nltk
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from ..caching_utils import cache_result, get_dataframe_hash
from ..hallucination_detection_abc import HallucinationDetectionMethod
from ..llm_base import LLMBase

nltk.download("punkt")


@cache_result(cache_dir="cache/sent_scores", message="Get sentence scores")
def get_perplexity_scores(
    logits: torch.Tensor, 
    answer_ids: torch.Tensor,
    min_k: Optional[float] = None,
    hash: Optional[str] = None,
    cache_name: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> list[torch.Tensor]:
    """Compute the perplexity of model predictions for given tokenized inputs.

    This function computes the perplexity by taking the negative log probability
    of the correct tokens and exponentiating the mean. If `min_k` is provided,
    it filters the lowest probabilities to compute a restricted perplexity.

    Args:
        logits: A list or array of model logits (samples x seq_len x vocab_size).
        tok_ins: A list of tokenized input IDs for each sample.
        tok_lens (list): A list of (start, end) indices specifying the portion of the
            sequence to evaluate.
        min_k (float, optional): A fraction of tokens to consider from the lowest
            probabilities. If not None, only these tokens are considered.

    Returns:
        np.array: An array of perplexity values for each sample.
    """
    perplexity_scores = []
    for logit, answer_id in zip(logits, answer_ids):
        logprob = torch.log(F.softmax(logit, dim=-1))
        answer_id = answer_id.unsqueeze(-1)
        perplexity = torch.gather(logprob, dim=-1, index=answer_id)
        perplexity = perplexity.squeeze(-1)

        if min_k is not None:
            perplexity = torch.topk(
                perplexity, 
                k=int(min_k * perplexity.shape[0]), 
                dim=-1, 
                largest=False
            )[0]
        perplexity = torch.exp(-perplexity.mean()).cpu().numpy()
        perplexity_scores.append(-perplexity)

    return np.array(perplexity_scores)

@cache_result(cache_dir="cache/token_distribution", message="Get token distributions")
def get_logits(
    X: pd.DataFrame,
    model: LLMBase,
    hash: Optional[str] = None,
    cache_name: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> list[torch.Tensor]:
    """Retrieve the distribution on each token of the model for each input in the DataFrame."""
    logits = []
    token_ids = []

    for output, prompt_ids, answer_ids in model.generate_llm_outputs(
        X, output_hidden_states=False, output_attentions=False
    ):
        len_answer = len(answer_ids[0])
        logits.append(output.logits[0, -len_answer:].cpu())
        token_ids.append(answer_ids["input_ids"].squeeze(0))

    return logits, token_ids


@dataclass
class Perplexity(HallucinationDetectionMethod):
    model_name: Literal["Llama-2-7b-chat-hf", "Mistral-7B-Instruct-v0.1"] = (
        "Llama-2-7b-chat-hf"
    )
    
    min_k: Optional[float] = None
    dtype: str = "float16"
    device: str = "cuda"
    cache_dir: str = "cache/perplexity_scores"
    scores_save_dir: str = "cache/generated_logits"

    
    """
    """

    def __post_init__(self):
        pass

    def fit(self, X: pd.DataFrame, y: pd.Series, *args) -> "Perplexity":
        """ """
        return self

    def transform(self, X: Any) -> List[List[float]]:
        """
        """

        llm = LLMBase(self.model_name, self.dtype, self.device)

        data_hash = get_dataframe_hash(X)

        cachefile_general_name = f"{self.model_name}_{data_hash}"

        generated_logits_cache_name = cachefile_general_name + "_logits.joblib"
        logits, answer_ids = get_logits(
            X, 
            llm,
            cache_name=generated_logits_cache_name, 
            cache_dir=self.cache_dir)
        
        sent_scores_cache_name = cachefile_general_name + "_perplexity.joblib"
        scores = get_perplexity_scores(
            logits,
            answer_ids, 
            self.min_k,
            cache_name=sent_scores_cache_name,
            cache_dir=self.cache_dir,
        )

        return scores

    def predict_score(self, X: Any) -> np.ndarray[float]:
        """"""
        return np.array(X)