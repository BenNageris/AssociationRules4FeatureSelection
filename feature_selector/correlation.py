from scipy import stats

import pandas as pd


def chi_squared(df: pd.DataFrame, categorical_columns: list, target_column: str):
    chi_res = []
    for index, column_name in enumerate(categorical_columns):
        try:
            contingency_table = pd.crosstab(df[column_name], df[target_column])
            c, p, _, _ = stats.chi2_contingency(contingency_table)
        except:
            c = None
            p = None
        chi_res.append({'col1': column_name, 'col2': target_column, 'score': c, 'p_val': p})
    chi_df = pd.DataFrame(chi_res)
    return chi_df.sort_values('p_val', ascending=True)


def get_sorted_chi_squared_parameters(df: pd.DataFrame, categorical_columns: list, target_column: str):
    chi_squared_df: pd.DataFrame = chi_squared(
        df=df,
        categorical_columns=categorical_columns,
        target_column=target_column
    )
    return chi_squared_df['col1'].tolist()


def pearson(df: pd.DataFrame, columns: list, target_column: str):
    dtf_corr = df[columns].corr(method="pearson").loc[[target_column]]
    return dtf_corr
