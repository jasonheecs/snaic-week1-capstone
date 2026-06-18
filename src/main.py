"""main.py — entry point for the churn prediction pipeline.

Run (from project root):
    python src/main.py                     # uses default data/train.csv
    python src/main.py --train other.csv   # custom path

Pipeline
--------
1. load_data      — read CSVs from disk
2. DataCleaner    — drop id, cast types, split X / y
3. train_test_split — hold out a validation slice
4. ModelSelector  — cross-validate candidates, pick the best
5. Final fit      — retrain best pipeline on full training data
6. Evaluate       — ROC-AUC and F1 on the held-out test split
7. report         — print a summary table
"""
from __future__ import annotations

import argparse

import pandas as pd
from sklearn.model_selection import train_test_split

from data_cleaner import DataCleaner
from model_selector import ModelSelector

TRAIN_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_data(path: str) -> pd.DataFrame:
    """Read a CSV from disk and return it as a DataFrame.

    Parameters
    ----------
    path : path to the CSV file

    Returns
    -------
    pd.DataFrame — raw, unmodified rows from the file
    """
    raise NotImplementedError


def evaluate(pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    """Score a fitted pipeline on a held-out test set.

    Metrics: ROC-AUC and F1 (threshold = 0.5 by default).

    Parameters
    ----------
    pipeline : fitted sklearn Pipeline
    X_test   : feature DataFrame
    y_test   : binary target Series

    Returns
    -------
    dict with keys "roc_auc" and "f1"
    """
    raise NotImplementedError


def report(scores_table: pd.DataFrame, final_metrics: dict[str, float]) -> None:
    """Print a formatted summary of cross-validation results and final test scores.

    Parameters
    ----------
    scores_table   : DataFrame returned by ModelSelector.evaluate_all()
    final_metrics  : dict returned by evaluate()
    """
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a churn prediction model.")
    parser.add_argument("--train", default=TRAIN_PATH, help="Path to training CSV")
    args = parser.parse_args()

    try:
        # 1. Load
        raw = load_data(args.train)

        # 2. Clean
        cleaner = DataCleaner()
        X, y = cleaner.fit_transform(raw)

        # 3. Split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )

        # 4. Select best model via cross-validation
        selector = ModelSelector()
        scores_table = selector.evaluate_all(X_train, y_train)

        # 5. Refit best pipeline on full training split
        best_pipeline = selector.best()
        best_pipeline.fit(X_train, y_train)

        # 6. Final evaluation on held-out validation set
        final_metrics = evaluate(best_pipeline, X_val, y_val)

        # 7. Print report
        report(scores_table, final_metrics)

    except NotImplementedError:
        print("Pipeline not yet implemented — stub scaffolding only.")


if __name__ == "__main__":
    main()
