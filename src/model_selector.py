from __future__ import annotations

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
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

    # Registry of candidate models — four candidates spanning linear, bagging,
    # and boosting families (two boosting implementations).
    # Keys are display names; values are unfitted estimator instances.
    CANDIDATES: dict[str, object] = {
        # Family 1 — Linear: draws a straight decision boundary.
        "LogisticRegression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42
        ),
        # Family 2 — Bagging: builds many independent trees and averages votes.
        "RandomForest": RandomForestClassifier(
            class_weight="balanced", n_estimators=200, random_state=42, n_jobs=-1
        ),
        # Family 3 — Boosting: builds trees sequentially, each fixing the last.
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200, random_state=42
        ),
        # Family 3 (alt) — Boosting (histogram-based)
        "XGBoost": XGBClassifier(
            n_estimators=200,
            random_state=42,
            eval_metric="logloss",
            n_jobs=-1,
        ),
    }

    def __init__(self) -> None:
        self._champion: str | None = None

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
        rows = []

        for name, model in self.CANDIDATES.items():
            print(f"  [{name}] running {cv}-fold CV...", flush=True)

            # clone() gives cross_validate a fresh unfitted copy each fold
            cv_results = cross_validate(
                build_pipeline(model), X, y,
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
        self._champion = str(scores.loc[0, "model"])
        return scores

    @property
    def champion(self) -> str:
        """Name of the highest-scoring model from evaluate_all().

        Raises
        ------
        RuntimeError if evaluate_all() has not been called yet.
        """
        if self._champion is None:
            raise RuntimeError("Call evaluate_all() before champion.")
        return self._champion

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
        if self._champion is None:
            raise RuntimeError("Call evaluate_all() before best().")

        return build_pipeline(self.CANDIDATES[self._champion])
