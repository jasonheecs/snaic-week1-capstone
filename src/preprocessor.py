from __future__ import annotations

from functools import partial

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

# Column groups — imported by model_selector.py, do not redefine elsewhere.
MULTI_CLASS_COLS = [
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaymentMethod",
]
BINARY_COLS = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]


# ---------------------------------------------------------------------------
# Engineered features (for the ablation study — see src/ablation.py).
#
# Each is a *pure row-wise* function: it derives a new column from values in
# the same row only, with no statistics learned from the data. That property is
# what makes them leakage-safe — running them inside a cross-validation fold can
# never peek at other rows (the fixed tenure_bin edges below are hand-set, not
# fitted). They are applied by an optional FunctionTransformer step that runs
# *before* the ColumnTransformer, so the new column is available for encoding.
# ---------------------------------------------------------------------------
def _add_has_internet(df: pd.DataFrame) -> pd.DataFrame:
    """1 if the customer has any internet service, else 0.

    Collapses the six internet add-ons (OnlineSecurity ... StreamingMovies) —
    which are all locked to "No internet service" whenever InternetService is
    "No" — into a single flag, removing that structural redundancy.
    """
    df["HasInternet"] = (df["InternetService"] != "No").astype(int)
    return df


def _add_tenure_bin(df: pd.DataFrame) -> pd.DataFrame:
    """Bucket raw tenure (1-72 months) into new / mid / long-term bands.

    Churn risk is concentrated in the first few months then flattens, so a
    coarse band can capture that step better than the raw count. Edges are
    fixed (not derived from the data) to stay leakage-safe.
    """
    df["tenure_bin"] = pd.cut(
        df["tenure"], bins=[0, 6, 24, 72], labels=["new", "mid", "long"]
    ).astype(str)
    return df


def _add_charges_per_month(df: pd.DataFrame) -> pd.DataFrame:
    """Average spend per month of tenure (TotalCharges / tenure).

    A single rate feature that can stand in for the correlated trio
    tenure / MonthlyCharges / TotalCharges. clip(lower=1) guards against
    divide-by-zero (tenure never drops below 1 in practice).
    """
    df["ChargesPerMonth"] = df["TotalCharges"] / df["tenure"].clip(lower=1)
    return df


# Registry of engineered features: name -> (compute function, target encoder block).
# The block decides which encoder the new column is routed to in build_pipeline().
ENGINEERED: dict[str, tuple] = {
    "HasInternet":     (_add_has_internet,      "binary"),       # 0/1   -> OrdinalEncoder block
    "tenure_bin":      (_add_tenure_bin,        "multi_class"),  # bands -> OneHotEncoder block
    "ChargesPerMonth": (_add_charges_per_month, "numeric"),      # float -> StandardScaler block
}


def _apply_engineering(df: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """Return a copy of df with every engineered feature in `names` added.

    Copies first so the caller's DataFrame (and other CV folds) are never
    mutated. Used as the function behind the optional "engineer" pipeline step.
    """
    df = df.copy()
    for name in names:
        compute_fn, _ = ENGINEERED[name]
        df = compute_fn(df)
    return df


def build_pipeline(model, drop_cols=None, engineer=None) -> Pipeline:
    """Wrap a ColumnTransformer preprocessor and a classifier into one Pipeline.

    Preprocessing steps
    -------------------
    - MULTI_CLASS_COLS : OneHotEncoder — one dummy per category minus one (drop='first')
    - BINARY_COLS      : OrdinalEncoder — No→0 / Yes→1, Female→0 / Male→1 (alphabetical)
    - NUMERIC_COLS     : StandardScaler — zero mean, unit variance

    Parameters
    ----------
    model : an unfitted sklearn-compatible estimator
    drop_cols : list[str] | None
        Column names to exclude from the model. Because the ColumnTransformer
        uses remainder="drop", dropping is just removing the name from its
        encoder group — no need to touch the data itself. Used for ablation.
    engineer : list[str] | None
        Names of engineered features (keys of ENGINEERED) to add before
        encoding, e.g. ["HasInternet"]. Each is routed to the encoder block
        declared in the registry. Used for ablation.

    Returns
    -------
    sklearn.pipeline.Pipeline with steps [("preprocessor", ColumnTransformer), ("model", model)].
    When `engineer` is given, an extra ("engineer", FunctionTransformer) step is
    prepended; with the defaults (drop_cols=None, engineer=None) the pipeline is
    identical to the original two-step form.
    """
    drop = set(drop_cols or [])
    eng = list(engineer or [])

    # Start from the canonical groups, remove any dropped columns, then append
    # each engineered feature to the block its registry entry names.
    multi_class = [c for c in MULTI_CLASS_COLS if c not in drop]
    binary = [c for c in BINARY_COLS if c not in drop]
    numeric = [c for c in NUMERIC_COLS if c not in drop]
    for name in eng:
        block = ENGINEERED[name][1]
        {"multi_class": multi_class, "binary": binary, "numeric": numeric}[block].append(name)

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "ohe",
                OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
                multi_class,
            ),
            (
                "ord",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                binary,
            ),
            (
                # Scaling matters for LogisticRegression regularisation.
                # Tree models are scale-invariant so this is harmless for them.
                "scl",
                StandardScaler(),
                numeric,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    # Only add the engineering step when features are requested, so the default
    # call returns the exact two-step pipeline the unit tests pin down.
    steps = []
    if eng:
        steps.append(("engineer", FunctionTransformer(partial(_apply_engineering, names=eng))))
    steps += [("preprocessor", preprocessor), ("model", model)]
    return Pipeline(steps)
