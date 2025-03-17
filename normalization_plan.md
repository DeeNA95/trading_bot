# Normalization Plan

This document outlines the plan to modify the `normalise_ohlc` function in `data.py` to use a rolling window for normalization.

## 1. Normalization Explanation

The current `normalise_ohlc` function uses Z-score normalization, which transforms data to have a mean of 0 and a standard deviation of 1. This is beneficial for machine learning models as it ensures features with different scales contribute equally.

We will improve this by using a rolling window to calculate the mean and standard deviation. This means that for each data point, we'll calculate the mean and standard deviation of the preceding 'n' periods (the window size). This makes the normalization more adaptive to changing market conditions.

## 2. Modify `normalise_ohlc`

The function will be modified to use a rolling window. Here's a visual representation of the process:

```mermaid
graph LR
    A[Original Data] --> B(Calculate Rolling Mean);
    B --> C(Calculate Rolling Std);
    C --> D(Normalize: (Data - Rolling Mean) / Rolling Std);
    D --> E[Normalized Data];
```

## 3. Add a Parameter

A `window=20` parameter will be added to the function definition:

```python
def normalise_ohlc(self, df, window=20):
    # ...
```

This allows the user to control the window size, with a default of 20 periods.

## 4. Changes

The `normalise_ohlc` function will be modified to use `df['column'].rolling(window).mean()` and `df['column'].rolling(window).std()` instead of `df['column'].mean()` and `df['column'].std()`.

## 5. Consider Other Features

Consider normalizing other features in the DataFrame, such as technical indicators or risk metrics, for potentially improved model performance.

## 6. Data Leakage

It's important to use the same `window` parameter consistently during training, validation, and testing/inference to avoid data leakage. Data leakage occurs when information from the future is used to make predictions, leading to overly optimistic performance estimates.
