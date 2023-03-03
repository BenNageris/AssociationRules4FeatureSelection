from typing import List
import pandas as pd


def transform_one_hot_column(df: pd.DataFrame, columns: List, to_remove_one_hot_col=True):
    for column in columns:
        if df[column].value_counts().size <= 3:
            one_hot_cols: pd.DataFrame = pd.get_dummies(df[column], prefix=f"{column}_")
            if one_hot_cols.columns.size == 2 and to_remove_one_hot_col:
                # TODO: think whether to remove one of the one hot encoded columns
                one_hot_cols = one_hot_cols.drop(one_hot_cols.columns[0], axis=1)
            df = df.drop(column, axis=1)
            df = df.join(one_hot_cols)
    return df
