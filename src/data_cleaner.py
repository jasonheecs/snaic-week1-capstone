from __future__ import annotations

import pandas as pd


class DataCleaner:
    """Cleans the raw churn DataFrame and splits it into features and target.

    Usage
    -----
    cleaner = DataCleaner()
    X, y = cleaner.fit_transform(df)          # training set
    X_test, y_test = cleaner.transform(test)  # held-out set (uses same fit state)
    """

    def fit(self, df: pd.DataFrame) -> "DataCleaner":
        """Learn any state needed to clean unseen data (e.g. column presence).

        Parameters
        ----------
        df : raw DataFrame as loaded from train.csv

        Returns
        -------
        self
        """
        raise NotImplementedError

    def transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Apply cleaning steps to df.

        Steps (from EDA section 3.9):
        1. Drop the `id` column.
        2. Cast `SeniorCitizen` from int (0/1) to bool.
        3. Encode target `Churn` as 1 (Yes) / 0 (No); separate into y.
        4. Return (X, y) where X is the feature DataFrame.

        Parameters
        ----------
        df : raw DataFrame

        Returns
        -------
        X : pd.DataFrame  — feature matrix (no id, no Churn)
        y : pd.Series     — binary target (0/1)
        """
        raise NotImplementedError

    def fit_transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Fit and transform in one step. Equivalent to fit(df).transform(df)."""
        raise NotImplementedError
