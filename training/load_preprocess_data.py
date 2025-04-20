import pandas as pd
import numpy as np

def load_and_preprocess(file_path: str) -> pd.DataFrame:
    if file_path.endswith('.parquet'):
        df = pd.read_parquet(file_path)
    else:
        df = pd.read_csv(file_path)

    # Drop any unnamed index columns
    df = df.drop(columns=[col for col in df.columns if 'unnamed' in col.lower()], errors='ignore')

    # Process columns
    columns_to_drop = []
    for column in df.columns:
        col_lower = column.lower()

        if 'time' in col_lower:
            if 'open_time' in col_lower or 'close_time' in col_lower:
                try:
                    df[column] = pd.to_numeric(df[column], errors='coerce')
                except:
                    columns_to_drop.append(column)
            else:
                columns_to_drop.append(column)
        elif df[column].dtype == 'object':
            try:
                df[column] = pd.to_numeric(df[column], errors='coerce')
            except:
                columns_to_drop.append(column)

    # Clean up data
    df = df.drop(columns=columns_to_drop, errors='ignore')
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.ffill()

    original_len = len(df)
    df = df.dropna()
    new_len = len(df)

    if original_len > new_len:
        print(f"Dropped {original_len - new_len} rows containing NaNs")

    if df.isnull().values.any():
        raise ValueError("NaN values still present after cleaning")

    print(f"Final data shape: {df.shape}")
    print(f"Columns: {', '.join(df.columns)}")

    return df
