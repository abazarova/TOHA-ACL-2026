import os
from itertools import product
from pathlib import Path

import hydra
import yaml
from comet_ml import Experiment
from dotenv import load_dotenv
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from loguru import logger
from omegaconf import OmegaConf

from src.evaluation import evaluate
from src.evaluation.process_metrics import process_metrics
from src.methods.hallucination_detection_abc import HallucinationDetectionMethod, T
from src.preprocess.dataset_abc import HallucinationDetectionDataset

load_dotenv()


@hydra.main(version_base=None, config_path="config", config_name="redeep")
def main(cfg: OmegaConf):
    hydra_cfg = HydraConfig.get()
    preprocess_name: str = hydra_cfg.runtime.choices["preprocess"]
    method_name: str = hydra_cfg.runtime.choices["method"]
    model_name: str = cfg["model_name"]
    experiment_name = f"{preprocess_name}_{method_name}_{model_name}"

    experiment = Experiment(
        api_key=os.getenv("COMET_API_KEY"),
        project_name="llm-factuality",
    )

    # Set experiment name
    experiment.set_name(experiment_name)
    experiment.log_parameters(OmegaConf.to_container(cfg, resolve=True))

    dataset: HallucinationDetectionDataset = instantiate(cfg["preprocess"])
    X, y, train_idxs, test_idxs = dataset.process()

    model: HallucinationDetectionMethod = instantiate(cfg["method"], _convert_="all")

    assert method_name == "redeep", f"This method is not supported: {method_name}"

    metrics, best_model = evaluate(
        model=model,
        X=X,
        y=y,
        k=cfg["evaluation"]["k"],
        seed=cfg["evaluation"]["seed"],
        test_size=cfg["evaluation"]["test_size"],
        val_size=cfg["val_size"],
        tune_hyperparameters=cfg["tune_hyperparameters"],
    )

    table_str, raw_table = process_metrics(metrics, experiment)
    logger.success(f"Results for cross validation on {dataset.__class__.__name__}")
    print(table_str)
    experiment.log_metric("final_test_auroc", raw_table["roc_auc"].loc["test"]["mean"])
    experiment.end()


if __name__ == "__main__":
    main()
