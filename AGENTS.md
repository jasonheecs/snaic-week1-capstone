# AGENTS.md

This is a supervised machine learning pipeline built using sklearn

- Column groups (`MULTI_CLASS_COLS`, `BINARY_COLS`, `NUMERIC_COLS`) are defined in `src/preprocessor.py` — reuse, don't redefine
- Run from project root: `python src/main.py`
- `data/test.csv` is held out — do not use during development

## Conventions

- sklearn Pipeline pattern throughout; use `class_weight='balanced'` on all classifiers
- Call `DataCleaner.fit_transform` on training data and `.transform` only on test data
