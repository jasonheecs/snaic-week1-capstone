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
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from data_cleaner import DataCleaner
from model_selector import ModelSelector

TRAIN_PATH = "data/train.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_data(path: str) -> pd.DataFrame:
    """Read a CSV from disk and return it as a DataFrame."""

    return pd.read_csv(path)


def prepare_data(
    path: str = TRAIN_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load, split, and clean the data into train/validation features and targets.

    Bundles pipeline steps 1-3 so callers (main() and the ablation study) share
    one definition of the split and cleaning — guaranteeing they see identical
    data. The split is stratified on Churn with a fixed seed for reproducibility,
    and DataCleaner is fit on the train slice only (then applied to val) to avoid
    leaking validation information into preprocessing.

    Parameters
    ----------
    path : path to the training CSV (default data/train.csv)

    Returns
    -------
    (X_train, X_val, y_train, y_val)
    """
    raw = load_data(path)
    raw_train, raw_val = train_test_split(
        raw, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=raw["Churn"]
    )
    cleaner = DataCleaner()
    X_train, y_train = cleaner.fit_transform(raw_train)
    X_val, y_val = cleaner.transform(raw_val)
    return X_train, X_val, y_train, y_val


def evaluate(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
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
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = pipeline.predict(X_test)
    return {
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "f1": float(f1_score(y_test, y_pred)),
    }


def report(
    scores_table: pd.DataFrame,
    final_metrics: dict[str, float],
    champion: str,
    out_path: str = "results.txt",
) -> None:
    """Print a formatted summary of cross-validation results and final test scores.

    The same summary is also written to ``out_path`` so the run is persisted.

    Parameters
    ----------
    scores_table   : DataFrame returned by ModelSelector.evaluate_all()
    final_metrics  : dict returned by evaluate()
    champion       : name of the best model, from ModelSelector.champion
    out_path       : file to write the report to (default "results.txt")
    """
    lines = [
        "\n=== Cross-Validation Results ===",
        scores_table.to_string(index=False),
        f"\n=== Champion Model: {champion} ===",
        "\n=== Final Validation Metrics ===",
        *(f"  {k}: {v:.4f}" for k, v in final_metrics.items()),
    ]
    summary = "\n".join(lines)

    print(summary)
    with open(out_path, "w") as f:
        f.write(summary + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a churn prediction model.")
    parser.add_argument("--train", default=TRAIN_PATH, help="Path to training CSV")
    args = parser.parse_args()

    # 1-3. Load, split, and clean (see prepare_data for the leakage-safe details)
    X_train, X_val, y_train, y_val = prepare_data(args.train)

    # 4. Select best model via cross-validation
    selector = ModelSelector()
    scores_table = selector.evaluate_all(X_train, y_train)

    # 5. Refit best pipeline on full training split
    best_pipeline = selector.best()
    best_pipeline.fit(X_train, y_train)

    # 6. Final evaluation on held-out validation set
    final_metrics = evaluate(best_pipeline, X_val, y_val)

    # 7. Print report
    report(scores_table, final_metrics, selector.champion)


if __name__ == "__main__":
    main()
