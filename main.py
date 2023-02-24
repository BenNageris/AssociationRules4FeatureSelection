import preprocessing
from utils import load_datasets, get_min_support, get_min_confidence, get_target_column
from bin_columns import bin_numeric_columns
from association_rules import calc_apriori_rules, filter_rules_related_to_target

if __name__ == "__main__":
    dataset_name = "mobilePriceRange"
    # dataset_name = "HomeLoanApproval"

    train_df, test_df = load_datasets(dataset_name=dataset_name)

    train_df, columns_types = preprocessing.preprocessing(train_df)

    train_df = bin_numeric_columns(df=train_df, columns=columns_types.very_numerical)

    # apriori rules
    min_support = get_min_support(dataset_name=dataset_name)
    min_confidence = get_min_confidence(dataset_name=dataset_name)
    rules = calc_apriori_rules(train_df=train_df, min_support=min_support, min_confidence=min_confidence)

    target_column = get_target_column(dataset_name=dataset_name)
    print(f"target column:{target_column}")
    relevant_rules = filter_rules_related_to_target(rules=rules, target_column=target_column)

    for rule in relevant_rules:
        print(rule)
