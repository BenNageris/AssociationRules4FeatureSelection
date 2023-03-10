mobile_price_config = {
    "index_col": None,
    "target_column": "price_range",
    "min_confidence": 0.2,
    "min_support": 0.10,
}

home_loan_config = {
    "index_col": "Loan_ID",
    "target_column": "Loan_Status",
    "min_confidence": 0.5,
    "min_support": 0.4,
}

airlines_delay = {
    "index_col": "id",
    "target_column": "Class",
    "min_confidence": 0.07,
    "min_support": 0.07,
}

heart_attack = {
    "index_col": "id",
    "target_column": "output",
    "min_confidence": 0.3,
    "min_support": 0.3,
}

datasets_config = {
    "MobilePriceRange": mobile_price_config,
    "HomeLoanApproval": home_loan_config,
    "AirlinesDelay": airlines_delay,
    "HeartAttack": heart_attack,
}
