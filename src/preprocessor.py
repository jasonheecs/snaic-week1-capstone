from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


# Column groups come from EDA section 3.9 encoding recommendations.
MULTI_CLASS_COLS = [
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaymentMethod",
]
BINARY_COLS = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]


def build_pipeline(model) -> Pipeline:
    """Wrap a ColumnTransformer preprocessor and a classifier into one Pipeline.

    Preprocessing steps
    -------------------
    - MULTI_CLASS_COLS  : OneHotEncoder (drop='first', handle_unknown='ignore')
    - BINARY_COLS       : OrdinalEncoder (maps Yes→1 / No→0)
    - NUMERIC_COLS      : StandardScaler

    Parameters
    ----------
    model : an unfitted sklearn-compatible estimator (e.g. LogisticRegression())

    Returns
    -------
    sklearn.pipeline.Pipeline with steps:
        [("preprocessor", ColumnTransformer), ("model", model)]
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("ohe", OneHotEncoder(drop="first", handle_unknown="ignore"), MULTI_CLASS_COLS),
            ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), BINARY_COLS),
            # Scaling matters to regularize the data for LogisticRegression. Tree models
            # are scale-invariant, so this is harmless-but-unnecessary for them.
            ("scl", StandardScaler(), NUMERIC_COLS),
        ],
        remainder="drop",
    )
    return Pipeline([("preprocessor", preprocessor), ("model", model)])
