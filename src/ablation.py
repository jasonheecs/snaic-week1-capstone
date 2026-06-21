"""ablation.py — controlled ablation study on the champion model (XGBoost).

This is the tool for Part C of the capstone: measure how dropping columns (or
adding an engineered feature) changes the champion's cross-validated score.

Two ways to use it
-------------------
1. Batch (the main way) — edit the hardcoded EXPERIMENTS list below. Each entry
   is just a list of columns to drop (plus an optional engineered feature) and
   a one-line hypothesis. Then run the script to regenerate the full Ablation
   Log:

       python src/ablation.py        # writes ablation_results.txt

2. Interactive (optional) — import the wrapper and call it directly, as many
   times as you like with different lists. Data is loaded/cleaned once then
   cached, so repeat calls are fast:

       from ablation import run_ablation
       run_ablation(["gender", "PhoneService"])           # drop two columns
       run_ablation(drop_cols=INTERNET_ADDONS, engineer=["HasInternet"])

Everything is scored with 5-fold cross-validation on the TRAINING split only —
the held-out test set is never touched here (assignment rule). All experiments
reuse the exact champion config and pipeline from Parts A/B, so the all-features
baseline reproduces the ROC-AUC in results.txt.
"""
from __future__ import annotations

import warnings

import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import cross_validate

from main import prepare_data
from model_selector import ModelSelector
from preprocessor import (
    BINARY_COLS,
    ENGINEERED,
    MULTI_CLASS_COLS,
    NUMERIC_COLS,
    build_pipeline,
)

# The champion declared in Part B (results.txt). Pulling the estimator from
# ModelSelector keeps its hyperparameters in sync with the rest of the project.
CHAMPION = "XGBoost"
CV = 5  # 5-fold cross-validation, matching model selection in results.txt

# The six internet add-ons that are structurally locked to InternetService
# (all become "No internet service" when a customer has no internet). Handy as a
# ready-made drop list — e.g. collapse them into the HasInternet flag.
INTERNET_ADDONS = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]

# Every raw feature the pipeline knows about — used to catch typos in drop_cols.
_KNOWN_COLS = set(MULTI_CLASS_COLS + BINARY_COLS + NUMERIC_COLS)

# --- module-level caches (filled lazily on first use) ----------------------
_DATA: tuple | None = None          # (X_train, y_train), loaded + cleaned once
_BASELINE: dict[int, dict] = {}     # cv -> all-features metrics, the Δ reference


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _training_data() -> tuple[pd.DataFrame, pd.Series]:
    """Return the cleaned training features/target, loading them once and caching.

    Only the train split is used — ablation scores come from CV on training, so
    the validation/test slices are deliberately ignored here.
    """
    global _DATA
    if _DATA is None:
        X_train, _X_val, y_train, _y_val = prepare_data()
        _DATA = (X_train, y_train)
    return _DATA


def _clone_champion():
    """A fresh, unfitted copy of the champion estimator (params from Part B)."""
    return clone(ModelSelector.CANDIDATES[CHAMPION])


def _validate(drop_cols, engineer) -> None:
    """Guard against typos: warn on unknown drop names, error on unknown features."""
    unknown_drop = [c for c in (drop_cols or []) if c not in _KNOWN_COLS]
    if unknown_drop:
        warnings.warn(
            f"drop_cols not recognised and ignored: {unknown_drop}. "
            f"Known columns: {sorted(_KNOWN_COLS)}"
        )
    unknown_eng = [e for e in (engineer or []) if e not in ENGINEERED]
    if unknown_eng:
        raise KeyError(
            f"Unknown engineered feature(s): {unknown_eng}. "
            f"Available: {list(ENGINEERED)}"
        )


def _cv_metrics(drop_cols, engineer, cv) -> dict[str, float]:
    """Cross-validate the champion with the given feature change; return raw means/stds."""
    X, y = _training_data()
    pipeline = build_pipeline(_clone_champion(), drop_cols=drop_cols, engineer=engineer)
    res = cross_validate(pipeline, X, y, cv=cv, scoring=["roc_auc", "f1"])
    return {
        "roc_auc_mean": float(res["test_roc_auc"].mean()),
        "roc_auc_std":  float(res["test_roc_auc"].std()),
        "f1_mean":      float(res["test_f1"].mean()),
        "f1_std":       float(res["test_f1"].std()),
    }


def _baseline_metrics(cv: int = CV) -> dict[str, float]:
    """All-features champion metrics — the Δ reference, computed once per cv."""
    if cv not in _BASELINE:
        _BASELINE[cv] = _cv_metrics(None, None, cv)
    return _BASELINE[cv]


def _describe(drop_cols, engineer) -> str:
    """Short human label for an experiment when no explicit label is given."""
    parts = []
    if engineer:
        parts.append("+" + "+".join(engineer))
    if drop_cols:
        parts.append("drop " + ", ".join(drop_cols))
    return "; ".join(parts) or "baseline (all features)"


def _conclude(d_auc: float, baseline_std: float) -> str:
    """Draft conclusion from the AUC delta — a starting point to refine for the slide.

    "Significant" means the change moved ROC-AUC by more than roughly one
    baseline CV std, i.e. beyond the fold-to-fold noise floor.
    """
    if abs(d_auc) <= baseline_std:
        return "No significant change vs baseline - safe to drop for a simpler model."
    return (
        "Improves ROC-AUC - consider adopting."
        if d_auc > 0
        else "Lowers ROC-AUC - keep these features."
    )


# ---------------------------------------------------------------------------
# Public wrapper — @Team please call this!
# ---------------------------------------------------------------------------
def run_ablation(
    drop_cols=None,
    engineer=None,
    label: str | None = None,
    cv: int = CV,
    verbose: bool = True,
) -> dict:
    """Re-run the champion (XGBoost) with columns dropped and/or features added.

    Safe to call repeatedly with different lists — data and the baseline are
    cached after the first call.

    Parameters
    ----------
    drop_cols : list[str] | None
        Columns to exclude from the model, e.g. ["gender", "PhoneService"].
    engineer : list[str] | None
        Engineered features to add (keys of preprocessor.ENGINEERED),
        e.g. ["HasInternet"].
    label : str | None
        Display name for this run; auto-generated from the change if omitted.
    cv : int
        Number of cross-validation folds (default 5).
    verbose : bool
        Print a progress line and a one-line result summary.

    Returns
    -------
    dict — one row of metrics: experiment, dropped, added, roc_auc_mean/std,
    f1_mean/std, and d_auc (ROC-AUC change vs the all-features baseline).
    """
    _validate(drop_cols, engineer)
    name = label or _describe(drop_cols, engineer)
    if verbose:
        print(f"  [{name}] running {cv}-fold CV...", flush=True)

    baseline = _baseline_metrics(cv)
    # If nothing is dropped or added this run *is* the baseline — reuse it
    # instead of recomputing the same cross-validation.
    is_baseline = not drop_cols and not engineer
    metrics = baseline if is_baseline else _cv_metrics(drop_cols, engineer, cv)

    row = {
        "experiment": name,
        "dropped": len(drop_cols or []),
        "added": len(engineer or []),
        "roc_auc_mean": round(metrics["roc_auc_mean"], 4),
        "roc_auc_std": round(metrics["roc_auc_std"], 4),
        "f1_mean": round(metrics["f1_mean"], 4),
        "f1_std": round(metrics["f1_std"], 4),
        "d_auc": round(metrics["roc_auc_mean"] - baseline["roc_auc_mean"], 4),
    }
    if verbose:
        # Plain ASCII ("+/-", "delta") keeps output safe on every console/editor,
        # matching the existing results.txt style.
        print(
            f"  {row['experiment']:<42} "
            f"ROC-AUC {row['roc_auc_mean']:.4f} +/- {row['roc_auc_std']:.4f}  "
            f"(delta {row['d_auc']:+.4f})  "
            f"F1 {row['f1_mean']:.4f} +/- {row['f1_std']:.4f}"
        )
    return row


# ---------------------------------------------------------------------------
# EXPERIMENTS — edit this list, then run `python src/ablation.py`.
#
# How to add your own: append a dict with
#   name       : short title shown in the log
#   hypothesis : what you expect and why (one sentence)
#   drop_cols  : list of columns to drop (or [])
#   engineer   : list of engineered features to add (or [])  — options:
#                "HasInternet", "tenure_bin", "ChargesPerMonth"
# Part C asks for at most 4 experiments in the final deliverable; the four below
# are drawn straight from the EDA (eda.ipynb §4 univariate, §5 bivariate).
# ---------------------------------------------------------------------------
EXPERIMENTS: list[dict] = [
    {
        "name": "drop gender + PhoneService",
        "hypothesis": "gender (Cramer's V 0.007) and PhoneService (0.035) carry ~no churn "
                      "signal, so dropping them should not lower CV AUC and yields a simpler model.",
        "drop_cols": ["gender", "PhoneService"],
        "engineer": [],
    },
    {
        "name": "drop TotalCharges",
        "hypothesis": "TotalCharges is ~ tenure x MonthlyCharges (r~0.85); removing the "
                      "redundant numeric should leave CV AUC roughly unchanged.",
        "drop_cols": ["TotalCharges"],
        "engineer": [],
    },
    {
        "name": "collapse add-ons -> HasInternet",
        "hypothesis": "The six internet add-ons are structurally locked to InternetService; "
                      "one HasInternet flag should keep the signal with far fewer columns.",
        "drop_cols": INTERNET_ADDONS,
        "engineer": ["HasInternet"],
    },
    {
        "name": "lean combo (gender + PhoneService + TotalCharges)",
        "hypothesis": "Dropping all low-value/redundant features at once should give a "
                      "leaner model with CV AUC on par with the baseline.",
        "drop_cols": ["gender", "PhoneService", "TotalCharges"],
        "engineer": [],
    },
    # More feature-engineering examples (uncomment / adapt as needed):
    # {"name": "tenure_bin instead of raw tenure", "hypothesis": "...",
    #  "drop_cols": ["tenure"], "engineer": ["tenure_bin"]},
    # {"name": "ChargesPerMonth replaces charge trio", "hypothesis": "...",
    #  "drop_cols": ["tenure", "TotalCharges", "MonthlyCharges"], "engineer": ["ChargesPerMonth"]},
]


def _change_str(exp: dict) -> str:
    """Render an experiment's controlled change for the log table."""
    bits = []
    if exp["engineer"]:
        bits.append(f"add {exp['engineer']}")
    if exp["drop_cols"]:
        bits.append(f"drop {exp['drop_cols']}")
    return "; ".join(bits)


def write_log(rows: list[dict], experiments: list[dict], baseline: dict,
              out_path: str = "ablation_results.txt") -> None:
    """Print and save the Ablation Log (the Part C deliverable).

    Mirrors the print-and-write style of main.report(): a CV comparison table
    plus a per-experiment log of Hypothesis / Controlled change / CV impact /
    Conclusion.

    Parameters
    ----------
    rows        : metric rows from run_ablation — rows[0] is the baseline,
                  rows[1:] line up with `experiments`.
    experiments : the EXPERIMENTS definitions (for hypotheses and change text).
    baseline    : baseline metric row (for the header and the conclusion rule).
    out_path    : file to write (default ablation_results.txt).
    """
    table = pd.DataFrame(rows)

    lines = [
        f"\n=== Ablation Study  (Champion: {CHAMPION},  {CV}-fold CV on training set) ===",
        f"Baseline (all features):  "
        f"ROC-AUC {baseline['roc_auc_mean']:.4f} +/- {baseline['roc_auc_std']:.4f} | "
        f"F1 {baseline['f1_mean']:.4f} +/- {baseline['f1_std']:.4f}",
        "",
        table.to_string(index=False),
        "\n=== Ablation Log (Part C deliverable) ===",
    ]
    # rows[0] is the baseline; experiment rows start at index 1.
    for i, (exp, row) in enumerate(zip(experiments, rows[1:]), start=1):
        lines += [
            f"\n[{i}] {exp['name']}",
            f"    Hypothesis        : {exp['hypothesis']}",
            f"    Controlled change : {_change_str(exp)}",
            f"    CV impact         : "
            f"ROC-AUC {row['roc_auc_mean']:.4f} +/- {row['roc_auc_std']:.4f} "
            f"(delta {row['d_auc']:+.4f}) | F1 {row['f1_mean']:.4f} +/- {row['f1_std']:.4f}",
            f"    Conclusion        : {_conclude(row['d_auc'], baseline['roc_auc_std'])}",
        ]

    summary = "\n".join(lines)
    print(summary)
    # UTF-8 so the file stays valid even if a teammate puts non-ASCII in a hypothesis.
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(summary + "\n")


def main() -> None:
    """Run the baseline + every experiment in EXPERIMENTS and write the log."""
    # Baseline first so its CV is cached before the experiments compute their delta.
    baseline_row = run_ablation(label="baseline (all features)")
    rows = [baseline_row]
    for exp in EXPERIMENTS:
        rows.append(run_ablation(exp["drop_cols"], exp["engineer"], label=exp["name"]))

    write_log(rows, EXPERIMENTS, baseline_row)


if __name__ == "__main__":
    main()
