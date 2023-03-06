from typing import List
from sklearn import tree

from feature_selector import preprocessing
from feature_selector.feature_selection import feature_rank
from feature_selector.evaluation import evaluate_prediction
from feature_selector.bin_columns import bin_numeric_columns
from feature_selector.one_hot_column import transform_one_hot_column
from feature_selector.correlation import get_sorted_chi_squared_parameters
from feature_selector.association_rules import calc_apriori_rules, filter_rules_related_to_target
from feature_selector.evaluation import pre_process_df, preprocess_columns, split_to_features_and_target
from feature_selector.utils import load_datasets, get_min_support, get_min_confidence, get_target_column


if __name__ == "__main__":
    # dataset_name = "MobilePriceRange"
    # dataset_name = "HomeLoanApproval"
    # dataset_name = "AirlinesDelay"
    dataset_name = "HeartAttack"

    train_df, test_df = load_datasets(dataset_name=dataset_name)
    train_df = pre_process_df(train_df)

    train_df, columns_types = preprocessing.preprocessing(train_df)

    target_column = get_target_column(dataset_name=dataset_name)

    train_df = bin_numeric_columns(df=train_df, columns=columns_types.very_numerical)

    all_cols = columns_types.all()
    print(all_cols)
    all_cols.remove(target_column)
    train_df = transform_one_hot_column(df=train_df, columns=all_cols, to_remove_one_hot_col=False)

    # apriori rules
    min_support = get_min_support(dataset_name=dataset_name)
    min_confidence = get_min_confidence(dataset_name=dataset_name)
    rules = calc_apriori_rules(train_df=train_df, min_support=min_support, min_confidence=min_confidence)

    relevant_rules = filter_rules_related_to_target(rules=rules, target_column=target_column)

    feature_ranks = feature_rank(relevant_rules)
    print(f"feature ranks:{feature_ranks}")

    # TODO: create a tree classifier that uses the one-hot-encoder
    # chi squared
    train_df2, test_df2 = load_datasets(dataset_name=dataset_name)
    sorted_chi_squared_features_correlation: List = get_sorted_chi_squared_parameters(df=train_df2,
                                                                                      categorical_columns=all_cols,
                                                                                      target_column=target_column)
    print(sorted_chi_squared_features_correlation)
    parameters = [feature for feature, _ in feature_ranks]
    # parameters = ["ram", "sc_w", "battery_power", "int_memory", "fc", "dual_sim", "px_height"]
    # parameters = ["cp", "thall", "caa", "exng", "slp", "sex"]
    # parameters = ["Flight", "Time", "Length", "Airline", "AirportFrom", "AirportTo"]
    # parameters = ["Credit_History","Property_Area","Loan_Amount_Term","Married","Education","Gender","LoanAmount","Dependents","Unnamed: 0","ApplicantIncome","Self_Employed","CoapplicantIncome"]
    # parameters = ["thall", "cp", "caa", "exng", "slp", "sex", "restecg", "oldpeak", "chol", "age", "thalachh", "trtbps","fbs"]
    index = 8

    selected_features = parameters[:index]
    total_features = selected_features.copy()
    total_features.append(target_column)

    print(f"selected features:{selected_features}")
    print(f"total features:{total_features}")

    # Evaluation
    eval_train_df, eval_test_df = load_datasets(dataset_name=dataset_name)
    eval_train_df = pre_process_df(eval_train_df)
    eval_train_df, y_train_df = split_to_features_and_target(
        df=eval_train_df,
        relevant_features=selected_features,
        target_feature=target_column
    )
    eval_train_df, y_train_df = preprocess_columns(features_df=eval_train_df,
                                                   target_df=y_train_df,
                                                   columns_to_label_encode=columns_types.categorical)
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(eval_train_df, y_train_df)

    eval_train_df = pre_process_df(df=eval_test_df)
    eval_test_df, y_test_df = split_to_features_and_target(df=eval_test_df,
                                                           relevant_features=selected_features,
                                                           target_feature=target_column)

    eval_test_df, y_test_df = preprocess_columns(features_df=eval_test_df,
                                                 target_df=y_test_df,
                                                 columns_to_label_encode=columns_types.categorical)

    size = y_test_df.columns.size
    assert size == 1, "number of predicted columns is not 1"
    column_name = y_test_df.columns[0]
    test_values = y_test_df[column_name].tolist()
    predictions = clf.predict(eval_test_df)

    success_percentage = evaluate_prediction(predictions=predictions, test_values=test_values)
    print(f"success rate:{success_percentage}%")
