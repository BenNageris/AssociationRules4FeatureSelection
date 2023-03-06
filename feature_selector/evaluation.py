from typing import List
from sklearn import tree

import pandas as pd

from feature_selector.one_hot_column import transform_one_hot_column
from feature_selector.preprocessing import label_encode_columns


def create_decision_tree_model():
    """
    :return: tree decision tree classifier
    """
    return tree.DecisionTreeClassifier()


def fit_model(model: tree.DecisionTreeClassifier, train_df: pd.DataFrame, target_train_df: pd.DataFrame):
    """
    :param model: Decision tree Classifier
    :param train_df: pandas Dataframe
    :param target_train_df: pandas Dataframe
    :return: fits the model between the train and the target column
    """
    return model.fit(train_df, target_train_df)


def pre_process_df(df: pd.DataFrame):
    """
    :param df: pandas Dataframe
    :return: processed dataframe
    """
    df = df.dropna()
    return df


def split_to_features_and_target(df: pd.DataFrame, relevant_features, target_feature):
    """
    :param df: pandas Dataframe
    :param relevant_features: list of strings
    :param target_feature: string
    :return: returns a tuple constructed of the df with the relevant columns and the df with the target column
    """
    eval_train_df = df[relevant_features]
    y_train_df = df.copy()[[target_feature]]
    return eval_train_df, y_train_df


def preprocess_columns(features_df: pd.DataFrame, target_df: pd.DataFrame, columns_to_label_encode):
    """
    :param features_df: pandas Dataframe
    :param target_df: pandas Dataframe
    :param columns_to_label_encode: list of strings
    :return: Executes one-hot-col encoding to the two dfs and label encodes the column to label encode
    """
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
    """
    :param predictions: list of predictions
    :param test_values: list of real values
    :return: returns the share of correct predictions
    """
    assert len(predictions) == len(test_values), "prediction and target column sizes are not the same"
    count = 0
    for index, prediction in enumerate(predictions):
        if prediction == test_values[index]:
            count += 1
    return count / len(predictions)
