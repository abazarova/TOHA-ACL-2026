from .llm_utils import forward, wrapper
from .utils import copying_head_scores
from transformers.models.llama import modeling_llama
from transformers.models.mistral import modeling_mistral
from transformers.models.qwen2 import modeling_qwen2

from loguru import logger
from typing import List, Optional, Tuple
from tqdm import tqdm
from collections import defaultdict
from itertools import product
import optuna

import numpy as np
import pandas as pd
import torch
import gc
import torch.nn.functional as F
from dataclasses import dataclass
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from ..hallucination_detection_abc import HallucinationDetectionMethod

from ..llm_base import LLMBase
from ..customtypes import ModelName
from ..caching_utils import get_dataframe_hash, cache_result
from .utils import cache_redeep


@cache_result(cache_dir='cache/redeep/copy_heads', message='Look for Copy Heads')
def find_copy_heads(
    model: LLMBase,
    hash: Optional[str] = None,
    cache_name: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> List[Tuple[int, int]]:
    scores = copying_head_scores(model.llm)
    indicies_1d = np.argsort(scores)[:model.llm.config.num_attention_heads]
    num_layers = len(model.llm.model.layers)
    num_heads = model.llm.config.num_attention_heads
    indicies_2d = np.unravel_index(indicies_1d, shape=(num_layers, num_heads))
    return list(zip(*indicies_2d))

@cache_redeep(cache_dir="cache/redeep", message="Get ReDeEP scores")
def compute_redeep_scores(
    X: pd.DataFrame,
    model: LLMBase,
    copy_heads,
    knowledge_ffns,
    chunk_size = 8,
    hash: Optional[str] = None,
    cache_name: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> list[torch.Tensor]:
    """"""
    wrapper.reset()
    wrapper.post_init(model.llm)

    idxs = defaultdict(list)
    for l, h in copy_heads:
        idxs[l].append(h)

    knowledge_diff_batch = []
    external_similarity_batch = []
    
    for output, prompt_ids, answer_ids in tqdm(model.generate_llm_outputs(
        X, output_hidden_states=True, output_attentions=True
    ), total=len(X)):
        selected_attn = []
        for i, attn in enumerate(output.attentions):
            if i not in idxs.keys():
                continue
            selected_attn.append(attn[0, idxs[i]])

        if len(selected_attn) > 0:
            selected_attn = torch.cat(selected_attn, dim=0)
            last_hidden_layer = output.hidden_states[-1][0]
            
            num_heads = selected_attn.shape[0]
            prompt_len = len(prompt_ids[0])
            answer_len = len(answer_ids[0])
            k = max(1, int(prompt_len * 0.1))

            knowledge_diff = []
            external_similarity = []
            answer_positions = torch.arange(prompt_len, prompt_len + answer_len, device=selected_attn.device)
            for start in range(0, answer_len, chunk_size):
                end = min(start + chunk_size, answer_len)
                chunk_positions = answer_positions[start:end]  # (B,)
                pointer_probs = selected_attn[:, chunk_positions, :prompt_len]  # (H, B, P)
                top_k_indices = torch.topk(pointer_probs, k, dim=-1).indices  # (H, B, k)
                top_k_hidden_states = F.embedding(top_k_indices, last_hidden_layer)  # (H, B, k, D)
                attend_token_hidden_state = top_k_hidden_states.mean(dim=2)  # (H, B, D)
                current_hidden_states = last_hidden_layer[chunk_positions]  # (B, D)
                current_hidden_states = current_hidden_states.unsqueeze(0).expand(num_heads, -1, -1)  # (H, B, D)
                cos_sim = F.cosine_similarity(attend_token_hidden_state, current_hidden_states, dim=-1)  # (H, B)
                external_similarity.append(cos_sim)
            external_similarity = torch.cat(external_similarity, dim=1).T.cpu()  # (A, H)
        else:
            external_similarity = torch.tensor([])

        if len(knowledge_ffns) > 0:
            knowledge_diff = torch.stack([
                dist[len(prompt_ids):] for i, dist in enumerate(wrapper) if i in knowledge_ffns
            ], dim=1)
        else:
            knowledge_diff = torch.tensor([])

        wrapper.reset() 

        knowledge_diff_batch.append(knowledge_diff.float().numpy())
        external_similarity_batch.append(external_similarity.float().numpy())

    return external_similarity_batch, knowledge_diff_batch

@dataclass
class ReDeEP(HallucinationDetectionMethod):
    """"""
    model_name: ModelName
    dtype: str = "float16"
    device: str = "cuda"
    cache_dir: str = "cache/redeep"

    knowledge_ffns: List[int] = None
    copy_heads: List[Tuple[int, int]] = None
    top_k: int = None
    top_n: int = None
    alpha: float = 1.0
    beta: float = 0.2
    
    need_fit: bool = True
    chunk_size: int = 8

    def __post_init__(self):
        if self.model_name not in [
            "Llama-2-7b-chat-hf",
            "Llama-2-13b-chat-hf",
            "Mistral-7B-Instruct-v0.1",
            "Qwen2.5-7B-Instruct",
            "Llama-3.1-8B-Instruct",
        ]:
            raise ValueError(f"Unsupported model: {self.model_name}")
        
        modeling_llama.LlamaDecoderLayer.forward = forward
        modeling_mistral.MistralDecoderLayer.forward = forward
        modeling_qwen2.Qwen2DecoderLayer.forward = forward
        
        self.llm_model = LLMBase(self.model_name, self.dtype, self.device)
        
    def transform(self, X): 
        if self.llm_model.llm is None:
            self.llm_model.llm, self.llm_model.tokenizer = self.llm_model.instantiate_llm()
            self.llm_model.llm.config._attn_implementation = "eager"
        if self.copy_heads is None:
            self.copy_heads = self.find_copy_heads()
        if self.knowledge_ffns is None:
            self.knowledge_ffns = list(range(len(self.llm_model.llm.model.layers)))
        if self.top_n is None:
            self.top_n = len(self.copy_heads)
        if self.top_k is None:
            self.top_k = len(self.knowledge_ffns)

        data_hash = get_dataframe_hash(X)
        redeep_cache_name = f"{self.model_name}_{data_hash}_redeep.joblib"
        
        external_similarity, knowledge_diff = compute_redeep_scores(
            X=X,
            model=self.llm_model,
            copy_heads=self.copy_heads[:self.top_n],
            knowledge_ffns=self.knowledge_ffns[:self.top_k],
            chunk_size=self.chunk_size,
            cache_name=redeep_cache_name,
            cache_dir=self.cache_dir,
        )
        return list(zip(external_similarity, knowledge_diff))
    
    def fit(self, X, y, *args):
        return self
    
    def find_copy_heads(self):
        if self.llm_model.llm is None:
            self.llm_model.llm, self.llm_model.tokenizer = self.llm_model.instantiate_llm()
            self.llm_model.llm.config._attn_implementation = "eager"

        cache_name = f"{self.model_name}_copy_heads.joblib"
        idxs = find_copy_heads(self.llm_model, cache_name=cache_name, cache_dir=self.cache_dir)
        logger.info(f"The following copy heads will be used: {idxs}")
        return idxs
    
    def sort_copy_heads_and_ffns(self, X, y):
        external_similarity, knowledge_diff = [], []

        for es, kd in X:
            external_similarity.append(es.mean(axis=0))
            knowledge_diff.append(kd.mean(axis=0))

        external_similarity = np.array(external_similarity)
        knowledge_diff = np.array(knowledge_diff)

        ecs_roc_auc = [roc_auc_score(y, -head) for head in external_similarity.T]
        sorted_heads = sorted(range(len(ecs_roc_auc)), key=lambda x: ecs_roc_auc[x], reverse=True)

        pks_roc_auc = [roc_auc_score(y, layer) for layer in knowledge_diff.T]
        sorted_layers = sorted(range(len(pks_roc_auc)), key=lambda x: pks_roc_auc[x], reverse=True)

        return sorted_heads, sorted_layers

    def fit_hyperparameters(self, X_val: pd.DataFrame, y_val: pd.DataFrame):
        self.knowledge_ffns = None
        self.top_n = None
        self.top_k = None

        X_transformed = self.transform(X_val)

        sorted_heads, sorted_layers = self.sort_copy_heads_and_ffns(X_transformed, y_val)
        self.copy_heads = [self.copy_heads[idx] for idx in sorted_heads]
        self.knowledge_ffns = [self.knowledge_ffns[idx] for idx in sorted_layers]
        logger.info(f"Copy heads were ordered in the following way: {self.copy_heads}")
        logger.info(f"Knowledge FFNs were ordered in the following way: {self.knowledge_ffns}")

        external_similarity, knowledge_diff = [], []
        for es, kd in X_transformed:
            external_similarity.append(es[:, sorted_heads])
            knowledge_diff.append(kd[:, sorted_layers])
        X_transformed = list(zip(external_similarity, knowledge_diff))
        
        def objective(trial):
            top_n = trial.suggest_int("top_n", 1, len(self.copy_heads))
            top_k = trial.suggest_int("top_k", 1, len(self.knowledge_ffns))
            beta = trial.suggest_float("beta", 0.1, 2.0, step=0.1)

            self.top_n, self.top_k, self.alpha, self.beta = top_n, top_k, 1.0, beta

            external_similarity_trial, knowledge_diff_trial = [], []
            for es, kd in X_transformed:
                external_similarity_trial.append(es[:, :top_n])
                knowledge_diff_trial.append(kd[:, :top_k])

            pred_train = self.predict_score(list(zip(external_similarity_trial, knowledge_diff_trial)))
            roc_auc = roc_auc_score(y_val, pred_train)

            return roc_auc

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=1000)

        best_params = study.best_params

        # Set model parameters
        self.top_n = best_params["top_n"]
        self.top_k = best_params["top_k"]
        self.alpha = 1.0
        self.beta = best_params["beta"]
        logger.info(f"The following hyperparameters were selected: alpha = {self.alpha}, beta = {self.beta}, top_n = {self.top_n}, top_k = {self.top_k}")

    def aggregate_and_scale(self, X):
        external_similarity, knowledge_diff = [], []

        for es, kd in X:
            external_similarity.append(es.sum(axis=-1))
            knowledge_diff.append(kd.sum(axis=-1))

        es_min, es_max = float("inf"), -float("inf")
        kd_min, kd_max = float("inf"), -float("inf")

        for es, kd in zip(external_similarity, knowledge_diff):
            if es.min() < es_min:
                es_min = es.min()
            if es.max() > es_max:
                es_max = es.max()
            if kd.min() < kd_min:
                kd_min = kd.min()
            if kd.max() > kd_max:
                kd_max = kd.max()
            
        for i, (es, kd) in enumerate(zip(external_similarity, knowledge_diff)):
            external_similarity[i] = (es.mean() - es_min) / (es_max - es_min)
            knowledge_diff[i] = (kd.mean() - kd_min) / (kd_max - kd_min)

        external_similarity = np.array(external_similarity)
        knowledge_diff = np.array(knowledge_diff)

        features = np.stack((knowledge_diff, -external_similarity), axis=1)
        return features

    def predict_score(self, X) -> List[float]:
        features = self.aggregate_and_scale(X)

        pred = self.alpha * features[:, 0] + self.beta * features[:, 1]
        return pred
