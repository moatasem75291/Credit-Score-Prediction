import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


st.title("📊 Exploratory Data Analysis (EDA)")


def remove_underscore(sample: str) -> float:
    """Convert string number with underscores into float, removing invalid characters."""
    if pd.notnull(sample):
        return pd.to_numeric(str(sample).strip("_")[:20], errors="coerce")
    return np.nan


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
    "Credit_Utilization_Ratio",
    "Total_EMI_per_month",
    "Amount_invested_monthly",
    "Monthly_Balance",
    "Delay_from_due_date",
]
low_cardinality_cols = [
    "Credit_Mix",
    "Payment_of_Min_Amount",
    "Payment_Behaviour",
]
high_cardinality_cols = ["Occupation", "Num_Credit_Inquiries", "Credit_History_Age"]

uploaded_file = st.file_uploader("Upload your CSV dataset for analysis", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.write("### Dataset Overview")
    st.write("#### Preview")
    st.dataframe(df.head())
    st.write(f"#### Shape: {df.shape[0]} rows × {df.shape[1]} columns")

    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].apply(remove_underscore)

    st.write("### Missing Values")
    missing_values = df.isnull().sum()
    st.write(missing_values[missing_values > 0])

    st.write("### Summary Statistics")
    numeric_summary = df[numeric_cols].describe().T
    st.write(numeric_summary)

    st.write("### Categorical Features Analysis")
    for col in low_cardinality_cols:
        if col in df.columns:
            st.write(f"#### Distribution of `{col}`")
            fig, ax = plt.subplots()
            sns.countplot(x=df[col], ax=ax, order=df[col].value_counts().index)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
            st.pyplot(fig)

    st.write("### High-Cardinality Features Analysis")
    for col in high_cardinality_cols:
        if col in df.columns:
            st.write(f"#### Unique values in `{col}`: {df[col].nunique()}")

    st.write("### Correlation Heatmap")
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.write("### Key Insights")
    st.markdown(
        """
        - **Missing Values**: Some columns have significant missing data.
        - **Outliers**: Numeric columns like `Interest_Rate` and `Num_Bank_Accounts` show extreme values.
        - **Imbalanced Categories**: Columns like `Payment_of_Min_Amount` and `Credit_Mix` have dominant categories.
        - **High Cardinality**: Features like `Occupation` and `Type_of_Loan` need further cleaning or transformation.
        """
    )
else:
    st.info("Upload a dataset to start exploring!")
