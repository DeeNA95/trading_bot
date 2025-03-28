# Trading Bot Improvement Plan

This document outlines a phased plan to address the issues identified in `PROJECT_REVIEW_SUMMARY.md` and improve the trading bot towards institutional-grade standards.

## Phase 1: Foundational Fixes & Realistic Backtesting [COMPLETED]

**Goal:** Address critical flaws that invalidate current backtesting results and fix fundamental bugs.

1.  **[COMPLETED] Fix Backtesting Lookahead Bias (SL/TP):**
    *   **Action:** Modify `trading_env.py::_handle_backtesting_trade` (and `step` logic) to use *next* bar's data or simulated ticks to check SL/TP conditions *after* an action is decided. Ensure no current bar high/low information influences fills within the same bar.
    *   **File(s):** `rl_agent/environment/trading_env.py`
2.  **[COMPLETED] Implement Basic Slippage Model:**
    *   **Action:** Add a simple, configurable slippage model to the backtesting logic in `trading_env.py`. Start with fixed points or a volatility-based penalty.
    *   **File(s):** `rl_agent/environment/trading_env.py`
3.  **[COMPLETED] Correct Data Imputation:**
    *   **Action:** Remove `fillna(0)` in `train.py`. Implement `ffill().dropna()` for robust imputation.
    *   **File(s):** `train.py`
4.  **[COMPLETED] Fix Reward Function Bug:**
    *   **Action:** Correct the loss calculation logic in `reward.py::FuturesRiskAdjustedReward` to ensure losses are correctly penalized and amplified by leverage.
    *   **File(s):** `rl_agent/environment/reward.py`
5.  **[COMPLETED] Fix Agent Bugs:**
    *   **Action:** Ensure model instantiation occurs *before* optimizer creation in `ppo_agent.py`. Confirmed `evaluate` method uses deterministic policy mode.
    *   **File(s):** `rl_agent/agent/ppo_agent.py`
6.  **[COMPLETED] Set Random Seeds:**
    *   **Action:** Explicitly set seeds for `numpy`, `random`, and `torch` at the start of `train.py` for better reproducibility.
    *   **File(s):** `train.py`
7.  **[COMPLETED] Handle Initial NaNs in Environment:**
    *   **Action:** Addressed by using `dropna()` after `ffill()` in data preprocessing (Task 3). Ensures environment receives NaN-free data.
    *   **File(s):** `train.py` (via Task 3 fix)

## Phase 2: Enhancing Robustness & Validation

**Goal:** Improve the reliability of training and evaluation, add tracking, and refactor code.

7.  **Implement Walk-Forward Validation with Train/Val/Test Splits:**
    *   **Action:** Refactor `train.py` to use a walk-forward approach. Within each walk-forward step, split the available data into training, validation, and test sets. Use the validation set for hyperparameter optimization (HPO) and model selection (e.g., choosing the best epoch). Use the test set for final, unbiased performance evaluation (backtesting) of the selected model on unseen data.
    *   **File(s):** `train.py`, potentially `data.py` (for splitting logic)
8.  **Introduce Experiment Tracking:**
    *   **Action:** Integrate MLflow or Weights & Biases into the training pipeline (`train.py`, `ppo_agent.py`) to log parameters, metrics, code versions, and model artifacts.
    *   **File(s):** `train.py`, `ppo_agent.py`
9.  **Refactor & Centralize Logic:**
    *   **Action:** Move duplicated `_calculate_adaptive_leverage` logic to a shared utility or pass context appropriately. Review and potentially simplify `FuturesRiskAdjustedReward`.
    *   **File(s):** `rl_agent/environment/position_sizer.py`, `rl_agent/environment/trading_env.py`, `rl_agent/environment/reward.py`
10. **Improve Data Pipeline:**
    *   **Action:** Add data validation checks (gaps, outliers) in `data.py`. Ensure feature calculations avoid lookahead bias (calculate post-split or within env step).
    *   **File(s):** `data.py`, `train.py`, `rl_agent/environment/trading_env.py`
11. **Add Unit & Integration Tests:**
    *   **Action:** Create a `tests/` directory. Start adding `pytest` tests for critical functions (reward calculation, position sizing, data processing).
    *   **File(s):** New test files.
12. **Implement Conditional Model Saving:**
    *   **Action:** Modify the training loop (`train.py`, `ppo_agent.py`) to evaluate the model periodically on the *validation set*. Only save model checkpoints (e.g., best performing model based on validation reward/Sharpe ratio) if the performance meets a certain criterion (e.g., positive average reward, improvement over previous best). Log evaluation metrics to MLflow.
    *   **File(s):** `train.py`, `ppo_agent.py`

## Phase 3: Towards Institutional Standards

**Goal:** Implement more advanced features common in institutional systems.

12. **Sophisticated Slippage & Latency Modeling:**
    *   **Action:** Enhance backtester with liquidity-based slippage and simulated latency.
    *   **File(s):** `rl_agent/environment/trading_env.py`
13. **Hyperparameter Optimization (HPO):**
    *   **Action:** Integrate Optuna/Ray Tune for systematic HPO.
    *   **File(s):** `train.py` (and potentially helper scripts)
14. **Model Benchmarking:**
    *   **Action:** Implement simpler baseline models (LSTM, standard Transformer) and compare against PyraFormer. Explore alternative Transformer output usage (pooling).
    *   **File(s):** `rl_agent/agent/models.py`, `train.py`
15. **Parallel Training:**
    *   **Action:** Refactor to support parallel environments (e.g., using `stable-baselines3` or `Ray RLlib`).
    *   **File(s):** `train.py`, `ppo_agent.py`
16. **Data Redundancy & Quality:**
    *   **Action:** Explore secondary data sources. Implement more robust cleaning/validation.
    *   **File(s):** `data.py`
17. **Advanced Risk Management:**
    *   **Action:** Consider portfolio-level risk rules if applicable. Explore alternative sizing methods.
    *   **File(s):** `rl_agent/environment/position_sizer.py` (potentially new modules)
18. **Live Trading Robustness:**
    *   **Action:** (If pursuing live trading) Add comprehensive error handling, state reconciliation, monitoring. Reduce external API reliance.
    *   **File(s):** `rl_agent/environment/trading_env.py`, `rl_agent/environment/execution.py`
