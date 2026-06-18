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

        --- HOW fit() WORKS (sklearn pattern) ---

        fit() reads the training data and MEMORISES anything that a future
        transform() call needs to stay consistent.  It stores that information
        as attributes on `self` (e.g. self._expected_cols).

        Why does this matter?
          - You fit() ONCE on training data.
          - You then call transform() on BOTH the training set and the test set.
          - Because transform() uses the memorised state, it applies the exact
            same logic to both datasets.

        For a scaler example: fit() records the training mean & std; transform()
        subtracts that same mean and divides by that same std on any dataset.
        If you were to recompute them on the test set you'd be "leaking" test
        information into your preprocessing — a classic data-leakage bug.

        For THIS cleaner the cleaning steps are deterministic (drop a column,
        cast a type), so the main thing worth storing is the expected column
        list.  This lets transform() catch mismatched test files early with a
        clear error instead of a cryptic KeyError later.
        """
        # Store every column name present in the training DataFrame.
        # transform() will check the test set against this list.
        self._fit_columns = df.columns.tolist()
        return self  # returning self enables chaining: cleaner.fit(df).transform(df)

    def transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Apply cleaning steps to df.

        Steps (from EDA section 3.9):
        1. Drop the `id` column.
        2. Cast `SeniorCitizen` from int (0/1) to bool.
        3. Encode target `Churn` as 1 (Yes) / 0 (No); separate into y.
        4. Return (X, y) where X is the feature DataFrame.

        Note: Encoding categorical columns like Contract is typically done inside the sklearn Pipeline in ModelSelector
        
        
        Parameters
        ----------
        df : raw DataFrame

        Returns
        -------
        X : pd.DataFrame  — feature matrix (no id, no Churn)
        y : pd.Series     — binary target (0/1)

        --- HOW transform() WORKS ---

        transform() uses the state stored by fit() to process ANY dataset
        (train or test) identically.  It must never re-learn from the data it
        is transforming — only apply what fit() already learnt.
        """
        # Guard: make sure fit() was called first.
        if not hasattr(self, "_fit_columns"):
            raise RuntimeError("Call fit() before transform().")

        # Guard: catch column mismatches between train and test early.
        missing = set(self._fit_columns) - set(df.columns)
        if missing:
            raise ValueError(f"Columns present at fit time are missing: {missing}")

        # Work on a copy so the caller's DataFrame is never mutated.
        df = df.copy()

        # Step 1 — Drop the identifier column.
        # `id` is a row number with no predictive signal; keeping it would
        # let a model memorise row positions rather than learn real patterns.
        df = df.drop(columns=["id"])

        # Step 2 — Cast SeniorCitizen to bool.
        # The raw file stores 0/1 as integers.  Casting to bool makes the
        # meaning explicit and consistent with the other binary Yes/No columns
        # (which are strings and will be handled later by an encoder).
        df["SeniorCitizen"] = df["SeniorCitizen"].astype(bool)

        # Step 3 — Encode target and separate it from the features.
        # Machine-learning models need numbers, not strings.
        # map() replaces each string value with the corresponding integer;
        # any unexpected value becomes NaN (handy as a data-quality signal).
        y = df["Churn"].map({"Yes": 1, "No": 0})
        y.name = "Churn"

        # Step 4 — Build the feature matrix X by removing the target column.
        X = df.drop(columns=["Churn"])

        return X, y

    def fit_transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Fit and transform in one step. Equivalent to fit(df).transform(df).

        --- WHY THIS CONVENIENCE METHOD EXISTS ---

        On the training set you always do both steps together.
        fit_transform() is just shorthand — it calls fit() then transform()
        on the same DataFrame so you don't have to type both lines yourself.

        IMPORTANT: only call fit_transform() on TRAINING data.
        For validation / test data call transform() alone (no fit) so that
        the cleaner keeps using the state it learnt from training.
        """
        return self.fit(df).transform(df)
