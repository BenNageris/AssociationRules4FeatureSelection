from typing import List
from collections import Counter

import math
import efficient_apriori


def feature_rank(rules: List[efficient_apriori.rules.Rule], lift_difference: float = 0.05):
    """
    :param rules: List of rules
    :param lift_difference: float
    :return: returns a sorted tuple of feature and the number this feature occurs in the lhs of the rules
    """
    feature_counter = Counter()
    for rule in rules:
        if math.fabs(rule.lift - 1) < lift_difference:
            continue
        rule_lhs = rule.lhs
        for col_name, value in rule_lhs:
            column_name = col_name.split('__')[0]
            feature_counter.update([column_name])
    return feature_counter.most_common()
