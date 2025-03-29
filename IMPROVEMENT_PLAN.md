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

7.  **[COMPLETED] Implement Walk-Forward Validation with Train/Val/Test Splits & Conditional Saving:**
    *   **Action:** Refactor `train.py` to use a walk-forward approach. Within each walk-forward step, split data into training, validation, and test sets. Train on the training set. Periodically evaluate on the *validation set* and save the best model checkpoint based on validation performance (e.g., highest reward/Sharpe, meeting minimum criteria like positive reward). Use the *test set* only once per fold for final, unbiased performance evaluation of the best model selected via validation. Log validation and test metrics to MLflow.
    *   **File(s):** `train.py`, `ppo_agent.py`, potentially `data.py` (for splitting logic)
8.  **[COMPLETED] Introduce Experiment Tracking:**
    *   **Action:** Integrate MLflow into the training pipeline (`train.py`) to log parameters, metrics per fold, aggregated results, and model artifacts. MLflow server configured with Cloud SQL + GCS.
    *   **File(s):** `train.py`
9.  **[COMPLETED] Refactor & Centralize Logic:**
    *   **Action:** Moved duplicated `_calculate_adaptive_leverage` logic to `rl_agent/environment/utils.py`.
    *   **File(s):** `rl_agent/environment/position_sizer.py`, `rl_agent/environment/trading_env.py`, `rl_agent/environment/utils.py`
10. **Improve Data Pipeline & Refactor `data.py`:**
    *   **Action:** Add data validation checks (gaps, outliers). Ensure feature calculations avoid lookahead bias (calculate post-split or within env step). Remove `calculate_fractal_dimension`. Vectorize `calculate_price_density` for performance. Improve modularity by extracting feature calculations. Externalize hardcoded parameters.
    *   **File(s):** `data.py`, `train.py`
11. **Explore Unsupervised Feature Extraction:**
    *   **Action:** Research and potentially implement unsupervised methods (e.g., Autoencoders, PCA, clustering on returns/volatility) on the training data folds to extract alternative features. Evaluate if these features improve agent performance when added to the observation space.
    *   **File(s):** Potentially new feature engineering scripts, `data.py`, `train.py`
12. **Incorporate Alternative Data (e.g., X Sentiment):**
    *   **Action:** Investigate sources for historical crypto sentiment data derived from X (formerly Twitter). Develop methods to fetch, clean, align (time-synchronize), and integrate this sentiment data as additional features into the observation space. Evaluate its impact on model performance.
    *   **File(s):** Potentially new data fetching/processing scripts, `data.py`, `train.py`
13. **Add Unit & Integration Tests:**
    *   **Action:** Create a `tests/` directory. Start adding `pytest` tests for critical functions (reward calculation, position sizing, data processing).
    *   **File(s):** New test files.

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
