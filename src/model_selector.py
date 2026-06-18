from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


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

    # Registry of candidate models.
    # Keys are display names; values are unfitted estimator instances.
    # Extend this dict to add new candidates.
    CANDIDATES: dict[str, object] = {
        "LogisticRegression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42
        ),
        # "RandomForest": RandomForestClassifier(...),
        # "XGBoost": XGBClassifier(...),
        # "LightGBM": LGBMClassifier(...),
    }

    def evaluate_all(self, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> pd.DataFrame:
        """Cross-validate every candidate in CANDIDATES and return a scores table.

        For each model:
        1. Build a full pipeline via preprocessor.build_pipeline(model).
        2. Run cross_validate with scoring=["roc_auc", "f1"].
        3. Record mean ± std of each metric across folds.

        Parameters
        ----------
        X  : feature DataFrame (output of DataCleaner.transform)
        y  : binary target Series (0/1)
        cv : number of cross-validation folds

        Returns
        -------
        pd.DataFrame with columns: ["model", "roc_auc_mean", "roc_auc_std",
                                     "f1_mean", "f1_std"]
        sorted by roc_auc_mean descending.
        """
        raise NotImplementedError

    def best(self) -> Pipeline:
        """Return the pipeline for the highest-scoring model from the last evaluate_all call.

        Must be called after evaluate_all().

        Returns
        -------
        Unfitted sklearn Pipeline ready to call .fit(X, y) on.

        Raises
        ------
        RuntimeError if evaluate_all() has not been called yet.
        """
        raise NotImplementedError
