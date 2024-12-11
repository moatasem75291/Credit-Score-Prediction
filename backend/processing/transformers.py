from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np


def extract_Credit_History_Age(value):
    if isinstance(value, str):
        return int(value.split(" ")[0])


def remove_underscore(sample: str) -> float:
    """
    Function to convert the string number into a float,
    after removing the underscore characters.
    """
    if pd.notnull(sample):
        return pd.to_numeric(str(sample).replace("_", "")[:20])
    return np.nan


class RemoveUnderscore(BaseEstimator, TransformerMixin):
    """
    Transformer to remove underscores from numeric columns
    and convert them to float values.
    """

    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        if self.columns is None:
            self.columns = X.select_dtypes(include=["object", "string"]).columns
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            X[col] = X[col].apply(remove_underscore)
        return X


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Transformer for frequency encoding categorical columns with distribution-preserving NaN imputation."""

    def __init__(self, columns):
        self.columns = columns
        self.freq_maps = {}

    def fit(self, X, y=None):
        for col in self.columns:
            self.freq_maps[col] = X[col].value_counts(normalize=True).to_dict()
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            # Fill NaN values by sampling based on the original distribution
            value_counts = X[col].value_counts(normalize=True)
            distribution = value_counts.to_dict()
            fill_values = np.random.choice(
                list(distribution.keys()),
                size=X[col].isna().sum(),
                p=list(distribution.values()),
            )
            X.loc[X[col].isna(), col] = fill_values
            X[col] = X[col].map(self.freq_maps[col])
        return X


class ClipOutliers(BaseEstimator, TransformerMixin):
    """Transformer to clip outliers for numeric columns."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for col in X.columns:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            X[col] = np.clip(X[col], lower_bound, upper_bound)
        return X
