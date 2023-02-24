import pandas as pd


def bin_numeric_columns(df: pd.DataFrame, columns: list):
    train_df = df.copy()
    for c in columns:
        try:
            train_df[c] = pd.qcut(df[c], 5, labels=["very low", "low", "medium", "high", "very high"])
        except:
            train_df[c] = pd.cut(df[c], 5, labels=["very low", "low", "medium", "high", "very high"])
    return train_df
