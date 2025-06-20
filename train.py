import os
import sys
import json
import logging
from datetime import datetime
from typing import Optional, List

import typer
import optuna
from optuna.samplers import TPESampler
from optuna.storages import RDBStorage
from omegaconf import OmegaConf, DictConfig
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra

from training.trainer import Trainer
from training.model_factory import ModelConfig
from experiments.optuna_objective import optuna_objective

app = typer.Typer(help="RL Agent Training and Hyperparameter Optimization CLI", help_option_names=["-h", "--help"])

def load_config(overrides: Optional[List[str]] = None) -> DictConfig:
    """
    Load Hydra config with optional overrides.

    Args:
        overrides: List of Hydra CLI override strings.

    Returns:
        Composed Hydra dictionary config.
    """
    # Clear Hydra instance to allow reinitialization in multi-run environments
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    with initialize(config_path="conf", job_name="train_job"):
        cfg = compose(config_name="config", overrides=overrides or [])
    return cfg


@app.command(help="Run a single training fold with current configuration.")
def train(
    override: Optional[List[str]] = typer.Option(
        None, "--override", "-o", help="Override Hydra config entries, e.g. model.embedding_dim=512", show_default=False
    )
):
    """
    Run a training job with optionally overridden Hydra config parameters.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    logger = logging.getLogger("train")

    try:
        cfg = load_config(override)
        # Load and preprocess data
        from training.load_preprocess_data import load_and_preprocess

        train_data_path = cfg.env.train_data
        if not train_data_path or not os.path.isfile(train_data_path):
            logger.error(f"Training data file does not exist: {train_data_path}")
            raise FileNotFoundError(train_data_path)

        df = load_and_preprocess(train_data_path)

        # For simplicity, treat full dataset as train, val, test splits (can be adjusted for walk-forward)
        train_df = df
        val_df = df
        test_df = df

        model_cfg = ModelConfig(**OmegaConf.to_container(cfg.model, resolve=True))
        trainer_args = OmegaConf.to_container(cfg, resolve=True)

        trainer = Trainer(fold_num=0, train_df=train_df, val_df=val_df, test_df=test_df, model_config=model_cfg, trainer_args=trainer_args)
        metrics, _ = trainer.train_and_evaluate_fold()

        if metrics is None:
            logger.error("Training did not return any metrics.")
            raise RuntimeError("Training failed to produce metrics.")

        logger.info(f"Training completed. Metrics: {metrics}")
    except Exception as e:
        logger.exception("Training failed due to unexpected error.")
        sys.exit(1)


@app.command(help="Run hyperparameter optimization using Optuna (TPE sampler).")
def optimize(
    trials: int = typer.Option(30, "--trials", "-t", help="Number of Optuna trials to run."),
    study_name: str = typer.Option("ppo_hpo_study", "--study-name", "-s", help="Name of the Optuna study."),
    db_path: str = typer.Option("./optuna.db", "--db-path", "-d", help="SQLite DB file path for Optuna study."),
    output_dir: str = typer.Option("outputs", "--output-dir", "-o", help="Directory to save optimization results."),
):
    """
    Run an Optuna hyperparameter optimization study with the provided settings.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    logger = logging.getLogger("optuna.optimize")

    # Ensure output and DB directory exist
    os.makedirs(output_dir, exist_ok=True)
    db_dir = os.path.dirname(db_path)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir)

    storage_url = f"sqlite:///{os.path.abspath(db_path)}"
    sampler = TPESampler()

    # Create or load Optuna study
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=sampler,
        storage=storage_url,
        load_if_exists=True,
    )

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = os.path.join(output_dir, f"optuna_{run_timestamp}")
    os.makedirs(run_output_dir, exist_ok=True)

    logger.info(f"Starting Optuna study '{study_name}'. Results saved to {run_output_dir}")

    try:
        study.optimize(optuna_objective, n_trials=trials, show_progress_bar=True)
    except KeyboardInterrupt:
        logger.warning("Optimization interrupted by user.")

    best_trial = study.best_trial

    logger.info(f"Best trial #{best_trial.number} finished with value: {best_trial.value}")
    logger.info(f"Best hyperparameters: {best_trial.params}")

    # Save best hyperparameters to JSON
    best_params_path = os.path.join(run_output_dir, "best_params.json")
    with open(best_params_path, "w") as f:
        json.dump(best_trial.params, f, indent=4)
    logger.info(f"Saved best hyperparameters to {best_params_path}")

    # Save all trials results to CSV
    import pandas as pd

    trials_df = study.trials_dataframe()
    all_trials_path = os.path.join(run_output_dir, "all_trials.csv")
    trials_df.to_csv(all_trials_path, index=False)
    logger.info(f"Saved all trial results to {all_trials_path}")


if __name__ == "__main__":
    app()
