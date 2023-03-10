import pandas as pd

from feature_selector.datasets_config import datasets_config
from feature_selector.consts import TRAIN_DATASET, TEST_DATASET


def _is_dataset_conf_exists(dataset_name: str):
    """
    :param dataset_name: str
    :return: returns whether the dataset exists
    """
    return dataset_name in datasets_config.keys()


def get_dataset_conf(dataset_name: str):
    """
    :param dataset_name: str
    :return: returns a dict representing the dataset configuration
    """
    if not _is_dataset_conf_exists(dataset_name=dataset_name):
        raise Exception(f"There is no existing dataset named {dataset_name}")
    return datasets_config[dataset_name]


def get_index_col(dataset_name: str):
    """
    :param dataset_name: str
    :return: returns the index column value of the dataset
    """
    conf = get_dataset_conf(dataset_name)
    return conf['index_col']


def get_min_confidence(dataset_name: str):
    """
    :param dataset_name: str
    :return: returns the min confidence value of the dataset
    """
    conf = get_dataset_conf(dataset_name)
    return conf['min_confidence']


def get_min_support(dataset_name: str):
    """
    :param dataset_name: str
    :return: returns the min support value of the dataset
    """
    conf = get_dataset_conf(dataset_name)
    return conf['min_support']


def get_target_column(dataset_name: str):
    """
    :param dataset_name: str
    :return: returns the target column value of the dataset
    """
    conf = get_dataset_conf(dataset_name)
    return conf['target_column']


def load_datasets(dataset_name: str):
    """
    :param dataset_name: str
    :return: two dataframes which the first one is the training dataset and the second is the testing dataset
    """
    train_df = pd.read_csv(TRAIN_DATASET.format(dataset_name), index_col=get_index_col(dataset_name))
    test_df = pd.read_csv(TEST_DATASET.format(dataset_name), index_col=get_index_col(dataset_name))
    return train_df, test_df
