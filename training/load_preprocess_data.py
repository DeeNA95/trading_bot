import pandas as pd
import numpy as np

def load_and_preprocess(file_path: str) -> pd.DataFrame:
    if file_path.endswith('.parquet'):
        df = pd.read_parquet(file_path)
    else:
        df = pd.read_csv(file_path)
    print(f"Initial data shape: {df.shape}")

    # Drop any unnamed index columns
    df = df.drop(columns=[col for col in df.columns if 'unnamed' in col.lower()], errors='ignore')
    df = df.drop(columns='close_time', errors='ignore')
    print(f"Shape after dropping unnamed columns: {df.shape}")

    # First handle object columns
    object_columns = df.select_dtypes(include=['object']).columns
    columns_to_drop = []

    for column in object_columns:
        try:
            df[column] = pd.to_numeric(df[column])
        except:
            columns_to_drop.append(column)

    print(f"Object columns to be dropped: {columns_to_drop}")
    df = df.drop(columns=columns_to_drop, errors='ignore')
    print(f"Shape after dropping object columns: {df.shape}")

    # Now handle NaN and inf values
    print("\nColumns with NaN values before inf replacement:")
    print(df.isna().sum()[df.isna().sum() > 0])

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.ffill()

    # Finally drop remaining NaN rows
    original_len = len(df)
    df = df.dropna()
    new_len = len(df)

    if original_len > new_len:
        dropped = original_len - new_len
        print(f"\nDropped {dropped} rows containing NaNs")
        if new_len == 0:
            print("\nWARNING: All rows were dropped! Analyzing last state of dataframe:")
            print("\nColumns with NaN values:")
            print(df.isna().sum()[df.isna().sum() > 0])

    print(f"\nFinal data shape: {df.shape}")
    print(f"Columns: {', '.join(df.columns)}")

    return df
