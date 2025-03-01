#!/usr/bin/env python3

from data import DataHandler
import pandas as pd
import os
from datetime import datetime


def main():
    """
    Add all available metrics to BTC data and save to a new file.
    """
    # Initialize DataHandler
    dh = DataHandler()

    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)

    # Load BTC data
    try:
        # Try to load from all_time file if it exists
        print("Loading BTC data from all-time file...")
        btc_data = pd.read_csv(
            "data/BTC_USD_all_time.csv", index_col=0, parse_dates=True
        )
    except FileNotFoundError:
        print("All-time data file not found. Fetching from source...")
        # Fall back to getting data from Yahoo Finance
        btc_data = dh.get_yahoo_data(
            ["BTC-USD"], start_date="2010-07-17", end_date=datetime.now()
        )
        # Save the raw data
        dh.save_data_to_csv(btc_data, "data/BTC_USD_all_time.csv")

    # Make sure the DataFrame index is named 'timestamp'
    if btc_data.index.name != "timestamp":
        btc_data.index.name = "timestamp"

    print(f"Data loaded, shape: {btc_data.shape}")

    # 1. Calculate technical indicators
    print("Calculating technical indicators...")
    btc_data_with_indicators = dh.calculate_technical_indicators(btc_data)
    print(f"Technical indicators added, shape: {btc_data_with_indicators.shape}")

    # 2. Calculate risk metrics
    print("Calculating risk metrics...")
    btc_data_with_risk = dh.calculate_risk_metrics(btc_data_with_indicators)
    print(f"Risk metrics added, shape: {btc_data_with_risk.shape}")

    # 3. Identify trade setups
    print("Identifying trade setups...")
    btc_data_with_setups = dh.identify_trade_setups(btc_data_with_risk)
    print(f"Trade setups added, shape: {btc_data_with_setups.shape}")

    # Save the dataset with technical indicators and setups (before ML preprocessing)
    print("Saving dataset with all indicators and metrics...")
    dh.save_data_to_csv(btc_data_with_setups, "data/BTC_USD_with_metrics.csv")

    # 4. Handle NaN values before ML preprocessing
    # First, forward-fill any NaN values that occur due to rolling windows
    btc_data_filled = btc_data_with_setups.copy()

    # Fill NaN values in a way that preserves most of the data
    # For technical indicators and metrics that use rolling windows, forward-fill is appropriate
    btc_data_filled = btc_data_filled.fillna(method="ffill")

    # For columns that still have NaNs (usually at the beginning of the series), use backward-fill
    btc_data_filled = btc_data_filled.fillna(method="bfill")

    # For any remaining NaNs in numeric columns, fill with appropriate values
    numeric_cols = btc_data_filled.select_dtypes(include=["float64", "int64"]).columns
    for col in numeric_cols:
        if col in [
            "signal_trend",
            "signal_momentum",
            "signal_volatility",
            "signal_strength",
        ]:
            # These are signals that should be 0 (neutral) when not available
            btc_data_filled[col] = btc_data_filled[col].fillna(0)
        elif "ratio" in col:
            # Ratios should be set to neutral values when not available
            btc_data_filled[col] = btc_data_filled[col].fillna(0)
        elif col == "trade_setup":
            # Trade setup should be 'none' when not available
            btc_data_filled[col] = btc_data_filled[col].fillna("none")
        elif "norm" in col:
            # Normalized values should be 0 when not available
            btc_data_filled[col] = btc_data_filled[col].fillna(0)

    # 5. Preprocess for ML using our cleaned data (normalized features)
    print("Preprocessing for ML...")
    # Modify the DataHandler's preprocess_data_for_ml method call to avoid dropna()
    # We'll do this by creating a modified version of the method here

    # Calculate returns if not already present
    if "returns" not in btc_data_filled.columns:
        btc_data_filled["returns"] = btc_data_filled["close"].pct_change().fillna(0)

    # Normalize price-based features if not already present
    price_cols = ["open", "high", "low", "close"]
    for col in price_cols:
        norm_col = f"{col}_norm"
        if norm_col not in btc_data_filled.columns:
            btc_data_filled[norm_col] = (
                btc_data_filled[col] / btc_data_filled["close"].shift(1) - 1
            )
            btc_data_filled[norm_col] = btc_data_filled[norm_col].fillna(0)

    # Log transform volume if not already present
    if (
        "volume" in btc_data_filled.columns
        and "volume_norm" not in btc_data_filled.columns
    ):
        vol_mean = btc_data_filled["volume"].rolling(window=20).mean()
        btc_data_filled["volume_norm"] = btc_data_filled["volume"] / vol_mean
        btc_data_filled["volume_norm"] = btc_data_filled["volume_norm"].fillna(
            1
        )  # Use 1 for missing (average volume)

    btc_data_final = btc_data_filled
    print(f"ML preprocessing complete, shape: {btc_data_final.shape}")

    # Save the final data with all metrics
    print("Saving complete dataset...")
    dh.save_data_to_csv(btc_data_final, "data/BTC_USD_complete.csv")

    # Print sample of columns to verify
    print("\nFinal dataset columns:")
    for i, col in enumerate(btc_data_final.columns):
        print(f"  {i+1}. {col}")

    print(f"\nComplete dataset saved to 'data/BTC_USD_complete.csv'")
    print(f"Total rows: {btc_data_final.shape[0]}")
    print(f"Total columns: {btc_data_final.shape[1]}")


if __name__ == "__main__":
    main()
