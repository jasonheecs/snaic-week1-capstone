Your Goal: Predict the likelihood of customer churn.

## Project Structure

```
snaic-week1-capstone/
├── data/
│   ├── train.csv          # training set
│   └── test.csv           # held-out test set (final evaluation only)
├── src/
│   ├── main.py            # entry point — run with: python src/main.py
│   ├── data_cleaner.py    # DataCleaner: drops id, casts types, encodes target
│   ├── preprocessor.py    # build_pipeline(): ColumnTransformer + model step
│   └── model_selector.py  # ModelSelector: cross-validates candidates, picks best
├── tests/                 # pytest unit tests (conftest.py puts src/ on the path)
├── eda.ipynb              # exploratory data analysis (sections 1–5)
└── README.md
```

## Evaluation Metrics

The classes are imbalanced — roughly 78% of customers stay (label 0) and 22%
churn (label 1). That imbalance drives every metric decision below.

- **ROC-AUC (primary — used for model selection)**
  The ROC curve plots the true positive rate against the false positive rate as
  you sweep the classification threshold from 0 to 1, and AUC is the area under
  it. Intuitively, it's the probability that the model assigns a higher churn
  score to a randomly chosen churner than to a randomly chosen non-churner —
  i.e. how well it *ranks* customers by risk. 0.5 is random guessing, 1.0 is
  perfect separation.
  We use it as the selection metric for two reasons. First, it's
  **threshold-independent**: it evaluates the underlying `predict_proba`
  scores, so we're not locked into the arbitrary 0.5 cutoff while comparing
  models. Second, it's **insensitive to the class ratio**, so it doesn't get
  inflated by the 78% majority the way accuracy does. For a churn early-warning
  system, ranking who's most at risk is the job, which is exactly what AUC
  measures.

- **F1 (secondary — sanity check at threshold 0.5)**
  The harmonic mean of precision and recall on the positive (churn) class.
  Precision = of the customers we flagged, how many actually churned; recall =
  of the customers who churned, how many we caught. The harmonic mean punishes
  lopsided trade-offs, so a model can't game it by maximizing one at the
  other's expense. Unlike AUC, F1 is computed at a fixed threshold, so it tells
  us how the model behaves once it's actually making yes/no decisions.

**Why not accuracy?** With a 78/22 split, a degenerate classifier that predicts
"no churn" for everyone scores ~78% accuracy while having zero recall on the
class we care about — it never identifies a single churner. Accuracy is
dominated by the majority class and would happily reward that useless model.
ROC-AUC and F1 both focus on performance on the minority (churn) class, which
is the business objective.

### Running the training pipeline

```bash
# from the project root
python src/main.py                    # uses data/train.csv by default
python src/main.py --train <path>     # custom training CSV
```

### Running unit tests

```bash
# from the project root
python -m pytest tests/              # run the full suite
python -m pytest tests/ -v           # verbose: one line per test
python -m pytest tests/test_preprocessor.py   # a single test file
```