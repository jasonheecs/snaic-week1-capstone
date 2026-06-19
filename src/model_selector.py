from __future__ import annotations

import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline

from preprocessor import build_pipeline


class ModelSelector:
    """Evaluates a registry of candidate models and surfaces the best one.

    Usage
    -----
    selector = ModelSelector()
    scores = selector.evaluate_all(X_train, y_train, cv=5)
    print(scores)
    pipeline = selector.best()
    pipeline.fit(X_train, y_train)
    """

    def __init__(self) -> None:
        self._best_name: str | None = None
        self._candidates: dict[str, object] | None = None

    def _build_candidates(self, y: pd.Series) -> dict[str, object]:
        """Build the candidate dict, computing class-imbalance weights from y.

        Three distinct algorithmic families (satisfies Part B requirement):
          1. LogisticRegression — linear family
          2. RandomForest       — bagging (parallel tree ensemble)
          3. GradientBoosting   — boosting (sequential tree ensemble)
        """
        neg, pos = (y == 0).sum(), (y == 1).sum()

        return {
            # Family 1 — Linear
            # class_weight='balanced' auto up-weights the minority churn class.
            "LogisticRegression": LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=42,
            ),
            # Family 2 — Bagging (parallel trees)
            # Builds many independent trees and averages their votes.
            "RandomForest": RandomForestClassifier(
                class_weight="balanced",
                n_estimators=200,
                random_state=42,
                n_jobs=-1,
            ),
            # Family 3 — Boosting (sequential trees)
            # Builds trees one at a time, each correcting the previous one's errors.
            "GradientBoosting": GradientBoostingClassifier(
                n_estimators=200,
                random_state=42,
            ),
        }

    def evaluate_all(self, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> pd.DataFrame:
        """Cross-validate every candidate and return a scores table.

        For each model:
        1. Builds a full preprocessing + model pipeline via build_pipeline().
        2. Runs 5-fold cross-validation scoring ROC-AUC and F1.
        3. Records mean ± std of each metric across folds.

        Parameters
        ----------
        X  : feature DataFrame — output of DataCleaner.fit_transform / transform
        y  : binary target Series (0 = no churn, 1 = churn)
        cv : number of cross-validation folds (default 5)

        Returns
        -------
        pd.DataFrame with columns:
            model, roc_auc_mean, roc_auc_std, f1_mean, f1_std
        sorted by roc_auc_mean descending.
        """
        self._candidates = self._build_candidates(y)
        rows = []

        for name, model in self._candidates.items():
            print(f"  [{name}] running {cv}-fold CV...", flush=True)

            # clone() gives cross_validate a fresh unfitted copy each fold
            cv_results = cross_validate(
                build_pipeline(clone(model)), X, y,
                cv=cv,
                scoring=["roc_auc", "f1"],
            )

            rows.append({
                "model":        name,
                "roc_auc_mean": round(cv_results["test_roc_auc"].mean(), 4),
                "roc_auc_std":  round(cv_results["test_roc_auc"].std(),  4),
                "f1_mean":      round(cv_results["test_f1"].mean(),       4),
                "f1_std":       round(cv_results["test_f1"].std(),        4),
            })

        scores = (
            pd.DataFrame(rows)
            .sort_values("roc_auc_mean", ascending=False)
            .reset_index(drop=True)
        )
        self._best_name = scores.iloc[0]["model"]
        return scores

    def best(self) -> Pipeline:
        """Return an unfitted pipeline for the highest-scoring model.

        Must be called after evaluate_all().

        Returns
        -------
        Unfitted sklearn Pipeline — call .fit(X_train, y_train) on it.

        Raises
        ------
        RuntimeError if evaluate_all() has not been called yet.
        """
        if self._best_name is None:
            raise RuntimeError("Call evaluate_all() before best().")

        return build_pipeline(self._candidates[self._best_name])
