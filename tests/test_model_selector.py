import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline

from model_selector import ModelSelector

SCORE_COLUMNS = ["model", "roc_auc_mean", "roc_auc_std", "f1_mean", "f1_std"]


@pytest.fixture
def stubbed_selector(monkeypatch):
    """ModelSelector whose evaluate_all is replaced with a no-op stub.

    The stub just records a known champion (and returns a one-row scores frame),
    so best() can be tested in isolation without running real cross-validation.
    """
    selector = ModelSelector()
    champion = "LogisticRegression"

    def fake_evaluate_all(X=None, y=None, cv=5):
        selector._champion = champion
        return pd.DataFrame([dict.fromkeys(SCORE_COLUMNS)]).assign(model=champion)

    # Assigning a plain function as an instance attribute shadows the bound
    # method, so the stub takes X, y, cv directly (no self).
    monkeypatch.setattr(selector, "evaluate_all", fake_evaluate_all)
    return selector, champion


def test_best_raises_before_evaluate_all():
    with pytest.raises(RuntimeError):
        ModelSelector().best()


def test_best_returns_unfitted_pipeline(stubbed_selector):
    selector, _ = stubbed_selector
    selector.evaluate_all()

    pipeline = selector.best()

    assert isinstance(pipeline, Pipeline)
    assert [name for name, _ in pipeline.steps] == ["preprocessor", "model"]
    # Returned pipeline is unfitted: predicting before fitting raises.
    with pytest.raises(NotFittedError):
        pipeline.predict(pd.DataFrame())


def test_best_name_matches_top_row(stubbed_selector):
    selector, _ = stubbed_selector
    scores = selector.evaluate_all()

    best_name = str(scores.loc[0, "model"])
    expected_model_cls = type(ModelSelector.CANDIDATES[best_name])
    assert isinstance(selector.best().named_steps["model"], expected_model_cls)
