import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from preprocessor import (
    BINARY_COLS,
    MULTI_CLASS_COLS,
    NUMERIC_COLS,
    build_pipeline,
)

# Fixture sizing. Each multi-class column is given CATEGORIES_PER_MULTICLASS
# distinct values; with OneHotEncoder(drop="first") that yields
# (CATEGORIES_PER_MULTICLASS - 1) columns per multi-class column.
N_ROWS = 24
CATEGORIES_PER_MULTICLASS = 3


# What this fixture builds:
#   A 24-row table (one row = one fake customer) with every column the pipeline
#   knows how to process. The values are made up but follow simple repeating
#   patterns so the table is identical on every test run.
#
#   - Multi-class columns (e.g. Contract, PaymentMethod): each cell is "cat0",
#     "cat1", or "cat2", cycling row by row. So every such column has exactly 3
#     possible values.
#   - Binary columns (e.g. gender, Partner): alternate "Yes" / "No" by row.
#     SeniorCitizen is also binary, but mirrors the cleaner's bool output
#     (True/False) rather than Yes/No.
#   - Numeric columns: simple increasing/cycling numbers (tenure 0-11,
#     MonthlyCharges 20-55, TotalCharges 100-330).
@pytest.fixture
def features():
    """A small, deterministic feature frame with every column build_pipeline expects."""
    rows = {}
    for col in MULTI_CLASS_COLS:
        rows[col] = [f"cat{i % CATEGORIES_PER_MULTICLASS}" for i in range(N_ROWS)]
    for col in BINARY_COLS:
        rows[col] = ["Yes" if i % 2 == 0 else "No" for i in range(N_ROWS)]
    rows["tenure"] = [i % 12 for i in range(N_ROWS)]
    rows["MonthlyCharges"] = [20.0 + (i % 8) * 5 for i in range(N_ROWS)]
    rows["TotalCharges"] = [100.0 + i * 10 for i in range(N_ROWS)]
    # SeniorCitizen is a binary column; mirror the cleaner's bool output, which
    # overrides the Yes/No default the BINARY_COLS loop set above.
    rows["SeniorCitizen"] = [bool(i % 2) for i in range(N_ROWS)]
    return pd.DataFrame(rows)


def expected_feature_count(df: pd.DataFrame) -> int:
    """Number of output columns build_pipeline's preprocessor should produce.

    Three stacked blocks:
      - one-hot of each multi-class column, drop="first"  -> (nunique - 1) each
      - one ordinal-encoded column per binary column      -> len(BINARY_COLS)
      - one scaled column per numeric column               -> len(NUMERIC_COLS)
    """
    one_hot = sum(df[col].nunique() - 1 for col in MULTI_CLASS_COLS)
    return one_hot + len(BINARY_COLS) + len(NUMERIC_COLS)


def get_preprocessor() -> ColumnTransformer:
    """The ColumnTransformer step from a freshly built pipeline."""
    return build_pipeline(LogisticRegression()).named_steps["preprocessor"]


def densify(matrix):
    """ColumnTransformer may return a sparse matrix; return a dense ndarray either way."""
    return matrix.toarray() if sp.issparse(matrix) else np.asarray(matrix)


def test_build_pipeline_wraps_model_in_two_step_pipeline():
    """build_pipeline returns a [preprocessor, model] Pipeline holding the exact estimator passed in."""
    model = LogisticRegression()

    pipeline = build_pipeline(model)

    assert isinstance(pipeline, Pipeline)
    assert [name for name, _ in pipeline.steps] == ["preprocessor", "model"]
    assert pipeline.named_steps["model"] is model
    assert isinstance(pipeline.named_steps["preprocessor"], ColumnTransformer)


def test_column_transformer_routes_each_group_to_its_encoder():
    """Each column group is routed to its named transformer, and unlisted columns are dropped."""
    preprocessor = get_preprocessor()

    routing = {name: cols for name, _, cols in preprocessor.transformers}  # type: ignore[attr-defined]

    assert routing == {
        "ohe": MULTI_CLASS_COLS,
        "ord": BINARY_COLS,
        "scl": NUMERIC_COLS,
    }
    assert preprocessor.remainder == "drop"  # type: ignore[attr-defined]


def test_transform_output_is_numeric_with_expected_shape(features):
    """Transforming yields an all-numeric matrix sized one-hot + ordinal + scaled."""
    # Arrange
    preprocessor = get_preprocessor()

    # Act
    transformed = densify(preprocessor.fit_transform(features))

    # Assert
    assert np.issubdtype(transformed.dtype, np.number)
    assert transformed.shape == (len(features), expected_feature_count(features))


def test_numeric_columns_are_standardized(features):
    """The scaled block (trailing NUMERIC_COLS columns) has ~zero mean and unit variance."""
    # Arrange
    preprocessor = get_preprocessor()

    # Act
    transformed = densify(preprocessor.fit_transform(features))
    numeric_block = transformed[:, -len(NUMERIC_COLS):]

    # Assert
    assert np.allclose(numeric_block.mean(axis=0), 0, atol=1e-9)
    assert np.allclose(numeric_block.std(axis=0), 1, atol=1e-9)


def test_unknown_categories_at_transform_time_do_not_crash(features):
    """Categories unseen during fit are tolerated (OHE handle_unknown + ordinal unknown_value)."""
    # Arrange: fit on known data, then feed a row with never-seen category values.
    preprocessor = get_preprocessor()
    preprocessor.fit(features)
    unseen_row = features.iloc[[0]].copy()
    unseen_row[MULTI_CLASS_COLS[0]] = "NEVER_SEEN"
    unseen_row[BINARY_COLS[0]] = "Maybe"

    with pytest.warns(UserWarning, match="unknown categories"):
        transformed = densify(preprocessor.transform(unseen_row))

    # Assert
    assert transformed.shape == (1, expected_feature_count(features))


def test_unlisted_columns_are_dropped(features):
    """remainder="drop": a column in no group does not appear in the output."""
    # Arrange
    preprocessor = get_preprocessor()
    with_extra = features.copy()
    with_extra["unrelated"] = range(len(features))

    # Act
    transformed = densify(preprocessor.fit_transform(with_extra))

    # Assert
    assert transformed.shape[1] == expected_feature_count(features)


# --- ablation support: drop_cols / engineer ---------------------------------
def test_drop_cols_removes_columns_from_routing_and_output(features):
    """drop_cols excludes the named columns from their encoder group and the output."""
    # Arrange — drop one binary and one numeric column.
    pipeline = build_pipeline(LogisticRegression(), drop_cols=["gender", "TotalCharges"])
    preprocessor = pipeline.named_steps["preprocessor"]
    routing = {name: cols for name, _, cols in preprocessor.transformers}  # type: ignore[attr-defined]

    # Assert — dropped columns are gone from routing, others untouched.
    assert "gender" not in routing["ord"]
    assert "TotalCharges" not in routing["scl"]
    assert routing["ohe"] == MULTI_CLASS_COLS  # multi-class group unaffected
    # No engineering requested, so the pipeline stays the plain two-step form.
    assert [name for name, _ in pipeline.steps] == ["preprocessor", "model"]

    # Act — two fewer output columns (one ordinal + one numeric dropped).
    transformed = densify(preprocessor.fit_transform(features))
    assert transformed.shape[1] == expected_feature_count(features) - 2


def test_engineer_adds_step_routes_feature_and_fits(features):
    """engineer prepends a step, routes the new feature, and the pipeline fits end-to-end."""
    # Arrange
    pipeline = build_pipeline(LogisticRegression(), engineer=["HasInternet"])

    # Assert — an extra "engineer" step is prepended and the feature is routed
    # to the ordinal block per the ENGINEERED registry.
    assert [name for name, _ in pipeline.steps] == ["engineer", "preprocessor", "model"]
    routing = {name: cols for name, _, cols in pipeline.named_steps["preprocessor"].transformers}  # type: ignore[attr-defined]
    assert "HasInternet" in routing["ord"]

    # Act/Assert — fitting flows engineer -> preprocessor -> model without error,
    # proving HasInternet is created before the ColumnTransformer needs it.
    y = [i % 2 for i in range(len(features))]
    pipeline.fit(features, y)


def test_apply_engineering_adds_columns_without_mutating(features):
    """The engineering helper returns new columns and never mutates the caller's frame."""
    from preprocessor import _apply_engineering

    out = _apply_engineering(features, ["HasInternet", "ChargesPerMonth"])

    assert {"HasInternet", "ChargesPerMonth"} <= set(out.columns)
    # Original frame is untouched (we worked on a copy).
    assert "HasInternet" not in features.columns
    assert "ChargesPerMonth" not in features.columns
