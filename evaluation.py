from typing import List
import pandas as pd
from sklearn import tree

from one_hot_column import transform_one_hot_column
from preprocessing import label_encode_columns


def create_decision_tree_model():
    return tree.DecisionTreeClassifier()


def fit_model(model: tree.DecisionTreeClassifier, train_df: pd.DataFrame, target_train_df: pd.DataFrame):
    return model.fit(train_df, target_train_df)


def pre_process_df(df: pd.DataFrame):
    df = df.dropna()
    return df


def split_to_features_and_target(df: pd.DataFrame, relevant_features, target_feature):
    eval_train_df = df[relevant_features]
    y_train_df = df.copy()[[target_feature]]
    return eval_train_df, y_train_df


def preprocess_columns(features_df: pd.DataFrame, target_df: pd.DataFrame, columns_to_label_encode):
    features_df = transform_one_hot_column(
        df=features_df,
        columns=features_df.columns,
        to_remove_one_hot_col=True
    )
    target_df = transform_one_hot_column(
        df=target_df,
        columns=target_df.columns,
        to_remove_one_hot_col=True
    )
    features_df = label_encode_columns(df=features_df, columns_to_encode=columns_to_label_encode)
    return features_df, target_df


def evaluate_prediction(predictions: List, test_values: List):
    assert len(predictions) == len(test_values), "prediction and target column sizes are not the same"
    count = 0
    for index, prediction in enumerate(predictions):
        if prediction == test_values[index]:
            count += 1
    return count / len(predictions)
