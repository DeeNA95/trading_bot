import os
import logging
from typing import Dict, Any, Optional
import optuna
from optuna.trial import TrialState
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf, DictConfig
import numpy as np
import torch

from training.trainer import Trainer
from training.model_factory import ModelConfig
from training.load_preprocess_data import load_and_preprocess
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s : %(levelname)s : %(message)s")

def _load_data_from_config(cfg: DictConfig) -> pd.DataFrame:
    """
    Load and preprocess data from config path.

    Args:
        cfg: Hydra DictConfig with at least env.train_data defined.

    Returns:
        preprocessed DataFrame.
    """
    data_path: str = cfg.env.train_data
    if not data_path or not os.path.isfile(data_path):
        raise FileNotFoundError(f"Training data file not found at: {data_path}")
    df = load_and_preprocess(data_path)
    return df

def _create_model_config_from_cfg(cfg: DictConfig) -> ModelConfig:
    """
    Convert model config part of Hydra config into ModelConfig dataclass.

    Args:
        cfg: Hydra DictConfig with model keys.

    Returns:
        ModelConfig dataclass instance.
    """
    # OmegaConf to dict then unpack to ModelConfig
    model_params: Dict[str, Any] = OmegaConf.to_container(cfg.model, resolve=True)
    # Rename keys if necessary to match ModelConfig attributes
    # The config keys should match ModelConfig field names exactly
    return ModelConfig(**model_params)

def optuna_objective(trial: optuna.Trial) -> float:
    """
    Optuna hyperparameter optimisation objective function for PPO RL training.

    Runs full walk-forward training with PPO agent for given trial hyperparameters,
    supports pruning and checkpoint resume.

    Args:
        trial: Optuna trial object to suggest hyperparameters and report progress.

    Returns:
        Mean validation reward averaged over all time series splits.
    """
    # Reset Hydra to avoid duplicate config state if running in notebook or multi-trial environment
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    # Initialize Hydra config with config directory root relative to this module
    with initialize(config_path="../conf", job_name="optuna_trial"):
        # Compose base config (all defaults from conf/config.yaml)
        cfg: DictConfig = compose(config_name="config")

        # Sample hyperparameters for PPO and model from Optuna trial and override Hydra config.
        # Example suggestions - user can adjust ranges here as needed or extend
        ppo_lr = trial.suggest_loguniform("ppo.lr", 1e-5, 1e-3)
        ppo_batch_size = trial.suggest_categorical("ppo.batch_size", [32, 64, 128])
        ppo_gamma = trial.suggest_uniform("ppo.gamma", 0.9, 0.999)
        model_embedding_dim = trial.suggest_categorical("model.embedding_dim", [128, 256, 512])
        model_n_encoder_layers = trial.suggest_int("model.n_encoder_layers", 2, 8)
        model_dropout = trial.suggest_uniform("model.dropout", 0.0, 0.3)

        # Create Hydra overrides dictionary
        overrides = {
            "ppo.lr": ppo_lr,
            "ppo.batch_size": ppo_batch_size,
            "ppo.gamma": ppo_gamma,
            "model.embedding_dim": model_embedding_dim,
            "model.n_encoder_layers": model_n_encoder_layers,
            "model.dropout": model_dropout,
        }

        # Merge overrides into original config
        cfg = OmegaConf.merge(cfg, OmegaConf.create(overrides))

        # Load and preprocess data once
        df = _load_data_from_config(cfg)

        # Perform walk-forward split train/val/test - simple time series splits based on cfg.train.n_splits
        n_splits: int = cfg.train.n_splits
        val_ratio: float = cfg.train.val_ratio
        episodes: int = cfg.train.episodes
        eval_freq: int = cfg.train.eval_freq
        save_path_base: str = cfg.train.save_path
        device: str = cfg.train.device

        # Checkpoint resume path - can be empty string or None
        checkpoint_path: Optional[str] = cfg.train.get("checkpoint_path", None)
        if checkpoint_path == "":
            checkpoint_path = None
        if checkpoint_path is not None and not os.path.isfile(checkpoint_path):
            logging.warning(f"Checkpoint path provided but file does not exist: {checkpoint_path}")
            checkpoint_path = None

        # Split data chronologically
        n_rows = len(df)
        split_size = n_rows // (n_splits + 1)
        val_scores = []

        for fold in range(n_splits):
            train_start = 0
            train_end = split_size * (fold + 1)
            val_start = train_end
            val_end = val_start + int(split_size * val_ratio)
            test_start = val_end
            test_end = split_size * (fold + 2) if (fold + 2) <= n_splits else n_rows

            train_df = df.iloc[train_start:train_end].reset_index(drop=True)
            val_df = df.iloc[val_start:val_end].reset_index(drop=True)
            test_df = df.iloc[test_start:test_end].reset_index(drop=True)

            # Prepare model config dataclass
            model_config = _create_model_config_from_cfg(cfg)

            # Prepare trainer_args dict from Hydra config PPO + env + train merged keys
            # Simplify the required args from cfg
            trainer_args = {
                "symbol": cfg.env.symbol,
                "leverage": cfg.env.leverage,
                "episodes": episodes,
                "batch_size": cfg.ppo.batch_size,
                "update_freq": getattr(cfg.ppo, "update_freq", 4),  # fallback update freq
                "eval_freq": eval_freq,
                "dynamic_leverage": cfg.env.dynamic_leverage,
                "use_risk_adjusted_rewards": cfg.env.use_risk_adjusted_rewards,
                "max_position": cfg.env.max_position,
                "balance": cfg.env.balance,
                "risk_reward": cfg.env.risk_reward,
                "stop_loss": cfg.env.stop_loss,
                "trade_fee": cfg.env.trade_fee,
                "save_path": os.path.join(save_path_base, f"optuna_trial_{trial.number}", f"fold_{fold}"),
                "device": device,
                # PPO params
                "lr": cfg.ppo.lr,
                "gamma": cfg.ppo.gamma,
                "gae_lambda": cfg.ppo.gae_lambda,
                "policy_clip": cfg.ppo.policy_clip,
                "n_epochs": cfg.ppo.n_epochs,
                "entropy_coef": cfg.ppo.entropy_coef,
                "value_coef": cfg.ppo.value_coef,
                "max_grad_norm": cfg.ppo.max_grad_norm,
                "use_gae": cfg.ppo.use_gae,
                "normalize_advantage": cfg.ppo.normalize_advantage,
                "weight_decay": cfg.ppo.weight_decay,
                # Passing checkpoint path only to fold 0 to avoid conflicts; this can be customized
                "checkpoint_path": checkpoint_path if fold == 0 else None,
            }

            logger.info(f"Starting training fold {fold + 1}/{n_splits} for trial {trial.number}...")

            trainer = Trainer(fold, train_df, val_df, test_df, model_config, trainer_args)
            metrics, _ = trainer.train_and_evaluate_fold()

            if metrics is None:
                logger.warning(f"No metrics returned for fold {fold}, pruning trial.")
                raise optuna.exceptions.TrialPruned()

            val_reward = metrics.get("avg_reward", np.nan)
            if np.isnan(val_reward):
                logger.warning(f"Fold {fold} returned NaN reward, pruning trial.")
                raise optuna.exceptions.TrialPruned()

            logger.info(f"Fold {fold} val reward: {val_reward:.3f}")

            # Report intermediate objective value to Optuna for pruning
            trial.report(val_reward, step=fold)

            # Prune if not promising
            if trial.should_prune():
                logger.info(f"Trial {trial.number} pruned at fold {fold}.")
                raise optuna.exceptions.TrialPruned()

            # Early stop if reward is very good
            if val_reward >= 1_000_000:
                logger.info(f"Trial {trial.number} reached reward >= 1,000,000 at fold {fold}, stopping early.")
                return val_reward

            val_scores.append(val_reward)

        mean_val_reward = float(np.mean(val_scores))
        logger.info(f"Trial {trial.number} completed with mean validation reward: {mean_val_reward:.3f}")
        return mean_val_reward
