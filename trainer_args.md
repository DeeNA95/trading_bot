# Consolidated List of Arguments for Training Initialization

This list covers arguments needed based on the structure in `train.py`, `training/trainer.py`, `training/model_factory.py`, `rl_agent/agent/ppo_agent.py`, and `rl_agent/environment/trading_env.py` after recent refactoring.

**I. Data & Environment Configuration (`BinanceFuturesCryptoEnv` & Data Loading)**

*   `train_data` (str): Path to the full historical data file (CSV or Parquet). *(From `train.py` argparse)*
*   `symbol` (str): Trading pair symbol (e.g., "ETHUSDT"). *(From `train.py` argparse, used by Env)*
*   `interval` (str): Data interval (e.g., '1m', '15m'). *(Currently only in `train.py` argparse, potentially needed for data loading/env context)*
*   `window_size` (int): Observation window size. *(Defined via `--window` in `train.py` argparse, used by Env and `ModelConfig`)*
*   `leverage` (int): Default trading leverage. *(From `train.py` argparse, used by Env)*
*   `max_position` (float): Maximum position size as a fraction of balance. *(From `train.py` argparse, used by Env)*
*   `balance` (float): Initial account balance. *(From `train.py` argparse, used by Env)*
*   `risk_reward` (float): Risk-reward ratio for SL/TP calculation (if used). *(From `train.py` argparse, used by Env)*
*   `stop_loss` (float): Stop loss percentage. *(From `train.py` argparse, used by Env)*
*   `trade_fee` (float): Trade fee percentage. *(From `train.py` argparse, used by Env)*
*   `dynamic_leverage` (bool): Whether to use dynamic leverage adjustment. *(From `train.py` argparse (`--static_leverage` flag inverts it), used by Env)*
*   `use_risk_adjusted_rewards` (bool): Whether to use the complex reward function. *(From `train.py` argparse (`--simple_rewards` flag inverts it), used by Env)*
*   *(Optional Env Params):* `max_leverage`, `indicators`, `mode` (set internally by Trainer), `base_url`, `margin_type`, `funding_rate_weight`, `liquidation_penalty_weight`, `open_interest_weight`, `volatility_lookback`, `data_fetch_interval`, `include_funding_rate`, `include_open_interest`, `include_liquidation_data`, `dry_run`, `slippage_fraction`. *(These have defaults in `BinanceFuturesCryptoEnv.__init__` but could potentially be exposed as args if needed)*

**II. Model Architecture Configuration (`ModelConfig`)**

*   `architecture` (str): Core transformer architecture type ("encoder\_only", "decoder\_only", "encoder\_decoder"). *(From `train.py` argparse)*
*   `embedding_dim` (int): Embedding dimension size. *(From `train.py` argparse)*
*   `n_encoder_layers` (int): Number of encoder layers (if applicable). *(From `train.py` argparse)*
*   `n_decoder_layers` (int): Number of decoder layers (if applicable). *(From `train.py` argparse)*
*   `dropout` (float): Dropout rate for model components. *(From `train.py` argparse)*
*   `attention_type` (str): Attention mechanism type ("mha", "mla", etc.). *(From `train.py` argparse)*
*   `n_heads` (int): Number of attention heads. *(From `train.py` argparse)*
*   `n_latents` (int, optional): Number of latents (for MLA). *(From `train.py` argparse)*
*   `n_groups` (int, optional): Number of groups (for GQA). *(From `train.py` argparse)*
*   `ffn_type` (str): Feed-forward network type ("standard", "moe"). *(From `train.py` argparse)*
*   `ffn_dim` (int, optional): Feed-forward hidden dimension. *(From `train.py` argparse)*
*   `n_experts` (int, optional): Number of experts (for MoE). *(From `train.py` argparse)*
*   `top_k` (int, optional): Top K experts to use (for MoE). *(From `train.py` argparse)*
*   `norm_type` (str): Normalization layer type ("layer\_norm"). *(From `train.py` argparse)*

*   **Residual Connection Configuration:**
    *   `residual_scale` (float): Scaling factor for residual connections (default: 1.0). *(From `ModelConfig`)*
    *   `use_gated_residual` (bool): Whether to use learnable gates for residual connections. *(From `ModelConfig`)*
    *   `use_final_norm` (bool): Whether to apply a final layer normalization after all residual connections. *(From `ModelConfig`)*

*   **Feature Extraction Configuration:**
    *   `feature_extractor_type` (str): Type of feature extractor architecture ("basic", "resnet", "inception"). *(From `train.py` argparse)*
    *   `feature_extractor_dim` (int): Hidden dimension for feature extractor. *(From `train.py` argparse)*
    *   `feature_extractor_layers` (int): Number of layers in feature extractor. *(From `train.py` argparse)*
    *   `use_skip_connections` (bool): Whether to use skip connections in feature extractor. *(From `train.py` argparse)*
    *   `use_layer_norm` (bool): Whether to use layer normalization instead of batch norm in feature extractor. *(From `train.py` argparse)*
    *   `use_instance_norm` (bool): Whether to use instance normalization in feature extractor. *(From `train.py` argparse)*
    *   `feature_dropout` (float): Dropout rate for feature extractor. *(From `train.py` argparse)*

*   **Actor-Critic Head Configuration:**
    *   `head_hidden_dim` (int): Hidden dimension for actor-critic heads. *(From `train.py` argparse)*
    *   `head_n_layers` (int): Number of layers in actor-critic heads. *(From `train.py` argparse)*
    *   `head_use_layer_norm` (bool): Whether to use layer normalization in actor-critic heads. *(From `train.py` argparse)*
    *   `head_use_residual` (bool): Whether to use residual connections in actor-critic heads. *(From `train.py` argparse)*
    *   `head_dropout` (float, optional): Dropout rate for actor-critic heads (None = use model dropout). *(From `train.py` argparse)*

*   *(Note): `window_size` is also part of `ModelConfig` but is typically derived from the environment arg (`--window`).*
*   *(Note): `action_dim` is part of `ModelConfig` but is determined by the environment's action space, not set via args.*

**III. PPO Hyperparameters (`PPOAgent`)**

*   `lr` (float): Learning rate. *(From `train.py` argparse)*
*   `gamma` (float): Discount factor. *(From `train.py` argparse)*
*   `gae_lambda` (float): GAE lambda parameter. *(From `train.py` argparse)*
*   `policy_clip` (float): PPO policy clipping parameter. *(From `train.py` argparse)*
*   `batch_size` (int): Batch size for PPO updates. *(From `train.py` argparse)*
*   `n_epochs` (int): Number of epochs per PPO update. *(From `train.py` argparse)*
*   `entropy_coef` (float): Entropy coefficient. *(From `train.py` argparse)*
*   `value_coef` (float): Value function coefficient. *(From `train.py` argparse)*
*   `max_grad_norm` (float): Max gradient norm for clipping. *(From `train.py` argparse)*
*   `use_gae` (bool): Whether to use Generalized Advantage Estimation. *(From `train.py` argparse)*
*   `normalize_advantage` (bool): Whether to normalize advantages. *(From `train.py` argparse)*
*   `weight_decay` (float): Adam weight decay. *(From `train.py` argparse)*

**IV. Training Loop / Walk-Forward Configuration (`Trainer` & `train.py` main)**

*   `episodes` (int): Number of training episodes per fold. *(From `train.py` argparse)*
*   `n_splits` (int): Number of splits for TimeSeriesSplit walk-forward validation. *(From `train.py` argparse)*
*   `val_ratio` (float): Fraction of training data per fold used for validation. *(From `train.py` argparse)*
*   `eval_freq` (int): Episodes between evaluations during training. *(From `train.py` argparse)*
*   `save_path` (str): Base path to save trained models and results. *(From `train.py` argparse)*
*   `device` (str): Device to run on ('auto', 'cpu', 'cuda', 'mps'). *(From `train.py` argparse)*
*   *(Note): `update_freq` is defined in `train.py` argparse but its usage might be implicit within `PPOAgent` based on `batch_size`.*
