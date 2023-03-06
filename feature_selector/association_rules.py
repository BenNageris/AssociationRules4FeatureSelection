from typing import List

import pandas as pd

from efficient_apriori import apriori


def get_transactions(df: pd.DataFrame):
    """
    :param df: pandas Dataframe
    :return: list of transactions
    """
    records = df.to_dict(orient='records')
    transactions = []
    for r in records:
        transactions.append(list(r.items()))
    return transactions


def filter_rules_related_to_target(rules: List, target_column: str):
    """
    :param rules: List
    :param target_column: str
    :return: return a list of rules which its rhs of the rule contain only the target column
    """
    relevant_rules = []
    for rule in rules:
        rhs = rule.rhs
        if len(rhs) > 1:
            continue
        column_name, value = rhs[0]
        column_name = column_name.split('__')[0]
        if column_name == target_column:
            relevant_rules.append(rule)
    return relevant_rules


def calc_apriori_rules(train_df: pd.DataFrame, min_support: float, min_confidence: float):
    """
    :param train_df: pandas Dataframe
    :param min_support: float
    :param min_confidence: float
    :return: List of rules that match the minimum support and confidence criteria
    """
    transactions = get_transactions(train_df)

    itemsets, rules = apriori(transactions, min_support=min_support, min_confidence=min_confidence)

    return rules

