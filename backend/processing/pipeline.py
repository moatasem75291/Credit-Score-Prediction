from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from processing.transformers import RemoveUnderscore, ClipOutliers

irrelevant_cols = [
    "ID",
    "Customer_ID",
    "SSN",
    "Name",
    "Month",
    "Type_of_Loan",
    "Payment_Behaviour",
]
numeric_cols = [
    "Age",
    "Annual_Income",
    "Monthly_Inhand_Salary",
    "Num_Bank_Accounts",
    "Num_Credit_Card",
    "Interest_Rate",
    "Num_of_Loan",
    "Num_of_Delayed_Payment",
    "Changed_Credit_Limit",
    "Outstanding_Debt",
    "Credit_History_Age",
    "Credit_Utilization_Ratio",
    "Total_EMI_per_month",
    "Amount_invested_monthly",
    "Monthly_Balance",
    "Delay_from_due_date",
]
low_cardinality_cols = ["Credit_Mix", "Payment_of_Min_Amount"]

high_cardinality_cols = ["Occupation", "Num_Credit_Inquiries"]

numeric_pipeline = Pipeline(
    steps=[
        ("remove_underscore", RemoveUnderscore(numeric_cols)),
        ("clip_outliers", ClipOutliers()),
        ("impute", SimpleImputer(strategy="median")),
        ("scale", MinMaxScaler()),
    ]
)


categorical_pipeline = Pipeline(
    steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ordinal", OrdinalEncoder()),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numeric_cols),
        ("low_card_cat", categorical_pipeline, low_cardinality_cols),
    ],
    remainder="passthrough",
)
