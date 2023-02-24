from typing import List
from dataclasses import dataclass

import pandas as pd


@dataclass
class ColumnsTypes:
    very_numerical: List
    categorical: List
    ordinals: List


def _split_columns_types_to_categories(train_df: pd.DataFrame):
    numeric_columns = train_df.dtypes[(train_df.dtypes == "float64") | (train_df.dtypes == "int64")].index.tolist()
    very_numerical = [nc for nc in numeric_columns if train_df[nc].nunique() > 20]
    categorical_columns = [c for c in train_df.columns if c not in numeric_columns]
    ordinals = list(set(numeric_columns) - set(very_numerical))
    return very_numerical, categorical_columns, ordinals


def _fill_numerical_na_values_with_column_mean(train_df: pd.DataFrame, numerical_columns):
    na_columns = train_df[numerical_columns].isna().sum()
    na_columns = na_columns[na_columns > 0]
    for nc in na_columns.index:
        train_df[nc].fillna(train_df[nc].mean(), inplace=True)
    return train_df


def _drop_categorical_columns_with_high_na_ratio(train_df: pd.DataFrame, categorical_columns, na_ratio: float = 0.7):
    nul_cols = train_df[categorical_columns].isna().sum() / len(train_df)
    drop_us = nul_cols[nul_cols > na_ratio]
    if len(drop_us.index) > 0:
        print(f"going to drop :{drop_us.index}")
    return train_df.drop(drop_us.index, axis=1), list(set(categorical_columns) - set(drop_us.index))


def preprocessing(train_df: pd.DataFrame):
    very_numerical, categorical_columns, ordinals = _split_columns_types_to_categories(train_df)
    train_df = _fill_numerical_na_values_with_column_mean(
        train_df=train_df,
        numerical_columns=very_numerical
    )
    train_df, categorical_columns = _drop_categorical_columns_with_high_na_ratio(
        train_df=train_df,
        categorical_columns=categorical_columns
    )
    train_df[categorical_columns] = train_df[categorical_columns].fillna('na')

    return train_df, ColumnsTypes(very_numerical=very_numerical, categorical=categorical_columns, ordinals=ordinals)
