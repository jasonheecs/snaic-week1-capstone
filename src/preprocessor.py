from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

# Column groups — imported by model_selector.py, do not redefine elsewhere.
MULTI_CLASS_COLS = [
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaymentMethod",
]
BINARY_COLS = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]
NUMERIC_COLS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]


def build_pipeline(model) -> Pipeline:
    """Wrap a ColumnTransformer preprocessor and a classifier into one Pipeline.

    Preprocessing steps
    -------------------
    - MULTI_CLASS_COLS : OneHotEncoder — one dummy per category minus one (drop='first')
    - BINARY_COLS      : OrdinalEncoder — No→0 / Yes→0, Female→0 / Male→1 (alphabetical)
    - NUMERIC_COLS     : StandardScaler — zero mean, unit variance

    Parameters
    ----------
    model : an unfitted sklearn-compatible estimator

    Returns
    -------
    sklearn.pipeline.Pipeline with steps [("preprocessor", ColumnTransformer), ("model", model)]
    """
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "ohe",
                OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
                MULTI_CLASS_COLS,
            ),
            (
                "ord",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                BINARY_COLS,
            ),
            (
                "num",
                StandardScaler(),
                NUMERIC_COLS,
            ),
        ],
        remainder="drop",  # silently drops any column not listed above (e.g. id if present)
        verbose_feature_names_out=False,
    )
    return Pipeline([("preprocessor", preprocessor), ("model", model)])
