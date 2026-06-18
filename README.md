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
├── eda.ipynb              # exploratory data analysis (sections 1–5)
└── README.md
```

### Running the training pipeline

```bash
# from the project root
python src/main.py                    # uses data/train.csv by default
python src/main.py --train <path>     # custom training CSV
```