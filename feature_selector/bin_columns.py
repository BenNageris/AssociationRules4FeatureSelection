import pandas as pd


def bin_numeric_columns(df: pd.DataFrame, columns: list):
    """
    :param df: pandas Dataframe
    :param columns: List of str
    :return: returns a dataframe which the columns were qcut or cut to 5 slices
    """
    train_df = df.copy()
    for c in columns:
        try:
            train_df[c] = pd.qcut(df[c], 5, labels=[1, 2, 3, 4, 5])
        except:
            train_df[c] = pd.cut(df[c], 5, labels=[1, 2, 3, 4, 5])
    return train_df
