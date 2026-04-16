import hashlib
import random
import typing as tp
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from tqdm import tqdm, trange

from haloscope.linear_probe import LinearClassifier
from haloscope.metric_utils import get_measures
from src.methods.caching_utils import get_dataframe_hash
from src.methods.extract_states import get_hidden_states
from src.methods.hallucination_detection_abc import HallucinationDetectionMethod
from src.methods.llm_base import LLMBase

from ..caching_utils import get_dataframe_hash


def save_model(model, best_layer, save_file):
    logger.info("==> Saving...")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "best_layer": best_layer,
        },
        save_file,
    )

@dataclass
class HaloscopeDetector(HallucinationDetectionMethod):
    """Implementation of the Haloscope hallucination detection method.

    This method works by:
    1. Extracting embeddings from LLM layers
    2. Using PCA/SVD to find directions that separate truthful vs hallucinated content
    3. Applying a linear or non-linear classifier on these projections

    Parameters
    ----------
    model_name : str
        Name of the model to use for embedding extraction
    cache_dir : str
        Directory to cache model embeddings
    k_components : int
        Number of principal components to use (default: 5)
    use_weighted_svd : bool
        Whether to weight the PCA components by singular values
    layer_range : tuple[int, int]
        Range of layers to test (min, max)
    device : str
        Device to run the model on ('cuda' or 'cpu')
    dtype : str
        Data type for model ('float16' or 'float32')
    is_centered : bool
        Whether to center the embeddings by subtracting the mean (default: True)

    """

    model_name: str
    cache_dir: str= "cache/haloscope"
    k_components: int = 5
    use_weighted_svd: bool = False
    layer_range: tuple[int, int] = (0, 32)  # Range of layers to test (min, max)
    device: str = "cuda"
    dtype: str = "float16"
    is_centered: bool = True  # Whether to center embeddings
    best_thr: tp.Optional[float] = None

    # Internal state
    _best_layer: int = field(default=None, init=False)
    _projection: np.ndarray = field(default=None, init=False)
    _best_sign: int = field(default=None, init=False)
    _mean_embedding: np.ndarray = field(default=None, init=False)
    _best_model: torch.nn.Module = field(default=None, init=False)
    _best_model_state_dict: dict[torch.Tensor] = field(default=None, init=False)

    def transform(self, X: pd.DataFrame) -> list[torch.Tensor]:
        """Extract last token embeddings for the best layer.

        If the model is not fitted yet, extract last token embeddings for all layers in the range.

        Parameters
        ----------
        X : pd.DataFrame
            Input dataframe with 'prompt' and 'response' columns

        Returns
        -------
        list[torch.Tensor]
            List of last token embeddings, where each element corresponds to a single sample
            If model is not fitted, each element is a tensor of shape [num_layers, hidden_dim]

        """
        logger.info(f"Transforming data for {self.__class__.__name__}")
        hash_ = get_dataframe_hash(X)
        self.save_path = Path(self.cache_dir) / f"{self.model_name}_{hash_}.pt"
        if self._best_model is None:
            try:
                cp = torch.load(self.save_path)
                self._best_model_state_dict = cp["state_dict"]
                self._best_layer = cp["best_layer"]
            except:
                logger.info("No saved state dict is found.")
        # Initialize LLM model once to be reused
        llm_model = LLMBase(self.model_name, self.dtype, self.device)

        # If model is not fitted, extract all layers in range
        # Get hidden states for each layer and extract last token embeddings
        layer_embeddings = []
        for layer in trange(self.layer_range[0], self.layer_range[1], desc="Getting hiddens for layer"):
            hiddens = self._get_hidden_for_layer(X, layer, llm_model)
            # Stack last token embeddings for each sample: [batch_size, hidden_dim]
            layer_embeddings.append(torch.stack([hidden[-1, :] for hidden in hiddens]))

        # Transpose to get samples-first structure
        # layer_embeddings: [num_layers, batch_size, hidden_dim]
        # Return: [batch_size, num_layers, hidden_dim]

        output = [
            torch.stack([layer_emb[i] for layer_emb in layer_embeddings])
            for i in range(len(layer_embeddings[0]))
        ]

        if self._best_model_state_dict is not None:
            self._best_model = LinearClassifier(layer_embeddings[0].shape[-1], 2)
            self._best_model.load_state_dict(self._best_model_state_dict)

        return output

    def _get_hidden_for_layer(
        self, X: pd.DataFrame, layer: int, llm_model: LLMBase
    ) -> list[torch.Tensor]:
        """Get hidden states for a specific layer.

        Parameters
        ----------
        X : pd.DataFrame
            Input data
        layer : int
            Layer to extract
        llm_model : LLMBase
            The LLM model instance to use

        Returns
        -------
        list[torch.Tensor]
            List of hidden states tensors, each with shape [seq_len, hidden_dim]

        """
        # Create cache name
        data_hash = get_dataframe_hash(X)
        cachefile_name = f"{self.model_name}_{data_hash}_{layer}_hiddens.joblib"

        # Get hidden states
        hiddens = get_hidden_states(
            X,
            layer,
            llm_model,
            cache_name=cachefile_name,
            cache_dir=self.cache_dir,
        )

        return hiddens

    def svd_embed_score(
        self, embed_generated_wild, gt_label, begin_k, k_span, mean=1, svd=1, weight=0
    ):
        embed_generated = embed_generated_wild
        best_auroc_over_k = 0
        best_layer_over_k = 0
        best_scores_over_k = None
        best_projection_over_k = None
        best_result = 0
        for k in tqdm(range(begin_k, k_span)):
            best_auroc = 0
            best_layer = 0
            best_scores = None
            mean_recorded = None
            best_projection = None
            for layer in range(len(embed_generated_wild[0])):
                if mean:
                    mean_recorded = embed_generated[:, layer, :].mean(0)
                    centered = embed_generated[:, layer, :] - mean_recorded

                else:
                    centered = embed_generated[:, layer, :]

                if not svd:
                    pca_model = PCA(n_components=k, whiten=False).fit(centered)
                    projection = pca_model.components_.T
                    mean_recorded = pca_model.mean_
                    if weight:
                        projection = pca_model.singular_values_ * projection
                else:
                    _, sin_value, V_p = torch.linalg.svd(
                        torch.from_numpy(centered).cuda()
                    )
                    projection = V_p[:k, :].T.cpu().data.numpy()
                    if weight:
                        projection = sin_value[:k] * projection

                scores = np.mean(np.matmul(centered, projection), -1, keepdims=True)
                assert scores.shape[1] == 1
                scores = np.sqrt(np.sum(np.square(scores), axis=1))

                measures1 = get_measures(
                    scores[gt_label == 1], scores[gt_label == 0], plot=False
                )
                measures2 = get_measures(
                    -scores[gt_label == 1], -scores[gt_label == 0], plot=False
                )

                if measures1[0] > measures2[0]:
                    measures = measures1
                    sign_layer = 1
                else:
                    measures = measures2
                    sign_layer = -1

                print(f"AUROC on layer {self.layer_range[0] + layer}: ", measures[0])
                if measures[0] > best_auroc:
                    best_auroc = measures[0]
                    best_result = [100 * measures[2], 100 * measures[0]]
                    best_layer = layer
                    best_scores = sign_layer * scores
                    best_projection = projection
                    best_mean = mean_recorded
                    best_sign = sign_layer
            print(
                "k: ",
                k,
                "best result: ",
                best_result,
                "layer: ",
                best_layer,
                "mean: ",
                mean,
                "svd: ",
                svd,
            )

            if best_auroc > best_auroc_over_k:
                best_auroc_over_k = best_auroc
                best_result_over_k = best_result
                best_layer_over_k = best_layer
                best_k = k
                best_sign_over_k = best_sign
                best_scores_over_k = best_scores
                best_projection_over_k = best_projection
                best_mean_over_k = best_mean

        return {
            "k": best_k,
            "best_layer": best_layer_over_k,
            "best_auroc": best_auroc_over_k,
            "best_result": best_result_over_k,
            "best_scores": best_scores_over_k,
            "best_mean": best_mean_over_k,
            "best_sign": best_sign_over_k,
            "best_projection": best_projection_over_k,
        }

    def fit(
        self, X_train: list[torch.Tensor], y_train: list[int], X_val: list[torch.Tensor], y_val: list[int],
    ) -> "HaloscopeDetector":
        """Fit the detector on training data.

        Parameters
        ----------
        X_train : list[torch.Tensor]
            List of samples, where each sample is a tensor of shape [num_layers, hidden_dim]
        y_train : list[int]
            Labels (1 for hallucination, 0 for truthful)

        Returns
        -------
        HaloscopeDetector
            Fitted detector

        """
        if self._best_model is None:
            logger.info(f"Fitting {self.__class__.__name__} on {len(X_train)} raw and {len(X_val)} annotated samples")

            X_train = torch.stack(X_train, dim=0).numpy()
            y_train = np.array(y_train)

            X_val = torch.stack(X_val, dim=0).numpy()
            y_val = np.array(y_val)

            self._results = self.svd_embed_score(
                X_val,
                y_val,
                1,
                11,
                mean=1,
                svd=0,
                weight=self.use_weighted_svd,
            )

            self._pca_model = PCA(n_components=self._results["k"], whiten=False).fit(
                X_train[:, self._results["best_layer"], :]
            )
            self._projection = self._pca_model.components_.T
            if self.use_weighted_svd:
                self._projection = self._pca_model.singular_values_ * self._projection
            scores = np.mean(
                np.matmul(X_train[:, self._results["best_layer"], :], self._projection),
                -1,
                keepdims=True,
            )
            assert scores.shape[1] == 1
            best_scores = (
                np.sqrt(np.sum(np.square(scores), axis=1)) * self._results["best_sign"]
            )

            thresholds = np.linspace(0, 1, num=20)[1:-1]

            best_auroc = 0
            for thres_wild in tqdm(thresholds):
                for layer in range(len(X_train[0])):
                    thres_wild_score = np.sort(best_scores)[
                        int(len(best_scores) * thres_wild)
                    ]
                    true_wild = X_train[:, layer, :][best_scores > thres_wild_score]
                    false_wild = X_train[:, layer, :][best_scores <= thres_wild_score]

                    embed_train = np.concatenate([true_wild, false_wild], 0)
                    label_train = np.concatenate(
                        [np.ones(len(true_wild)), np.zeros(len(false_wild))], 0
                    )

                    from haloscope.linear_probe import get_linear_acc

                    (
                        best_acc,
                        final_acc,
                        (clf, best_state, best_preds, preds, labels_val),
                        losses_train,
                    ) = get_linear_acc(
                        embed_train,
                        label_train,
                        embed_train,
                        label_train,
                        2,
                        epochs=50,
                        print_ret=True,
                        batch_size=512,
                        cosine=True,
                        nonlinear=False,
                        learning_rate=1e-3,
                        weight_decay=0.0003,
                    )

                    clf.eval()
                    output = clf(torch.from_numpy(X_val[:, layer, :]).float().cuda())
                    pca_wild_score_binary_cls = torch.sigmoid(output)

                    pca_wild_score_binary_cls = pca_wild_score_binary_cls.cpu().data.numpy()

                    roc_auc = roc_auc_score(y_val, pca_wild_score_binary_cls)
                    print(f"ROC_AUC, {thres_wild}:", roc_auc)
                    if roc_auc > best_auroc:
                        best_auroc = roc_auc
                        self._best_layer = layer
                        self._best_model = clf

            save_model(self._best_model, self._best_layer, self.save_path)
        return self

    def reset(self):
        """Return the model to the untrained state."""
        self._best_layer = None
        self._projection = None
        self._best_sign = None
        self._mean_embedding = None
        self._best_model = None
        self._best_model_state_dict = None

    def predict_score(self, X: list[torch.Tensor]) -> np.ndarray:
        """Predict hallucination scores for new samples.

        Parameters
        ----------
        X : list[torch.Tensor]
            List of samples, where each sample is a tensor of shape [num_layers, hidden_dim]

        Returns
        -------
        np.ndarray
            Array of hallucination scores, shape [batch_size]

        """
        self._best_model.to(self.device)
        X_ = torch.stack(X)[:, self._best_layer, :].float().to(self.device)

        scores = torch.sigmoid(self._best_model(X_)).detach().cpu().numpy()

        return scores
