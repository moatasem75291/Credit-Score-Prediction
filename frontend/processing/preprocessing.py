from sklearn.preprocessing import MinMaxScaler

from ..processing.transformers import FrequencyEncoder, extract_Credit_History_Age
from ..processing.pipeline import (
    preprocessor,
    irrelevant_cols,
    high_cardinality_cols,
)


def preprocess_data(df):
    """Preprocess the dataset with all steps combined."""

    df.drop(columns=irrelevant_cols, inplace=True)

    freq_encoder = FrequencyEncoder(columns=high_cardinality_cols)
    df = freq_encoder.fit_transform(df)

    df["Credit_History_Age"] = MinMaxScaler().fit_transform(
        df.Credit_History_Age.apply(extract_Credit_History_Age).values.reshape(-1, 1)
    )

    if "Credit_Score" in df.columns:
        credit_score_mapping = {"Standard": 0, "Poor": 1, "Good": 2}
        df["Credit_Score"] = df["Credit_Score"].map(credit_score_mapping)
        X, y = df.drop("Credit_Score", axis=1), df["Credit_Score"]
    else:
        X, y = df, None

    X_preprocessed = preprocessor.fit_transform(X)

    return X_preprocessed, y
