# ICT6001 Week01 Capstone — Task Split (Group of 5)

**Deadline:** 19 June 2026, 15:00  
**Deliverable:** Single 12-slide PDF report

---

## Parallel Execution Overview

```
PHASE 1 — All 5 work in parallel from the start
─────────────────────────────────────────────────────────────────────
M1  ██ EDA + feature engineering + sklearn pipeline ████████████
M2  ██ CV framework + 3 classifier stubs            ████░░░░░░░░  ← plug in M1's pipeline when ready
M3  ██ Ablation loop code + 4 hypothesis stubs      ████░░░░░░░░  ← run when M2 picks champion
M4  ██ Failure extraction code + data exploration   ████░░░░░░░░  ← fill results after M2 has val predictions
M5  ██ Slide deck skeleton + business analysis      ████████████  ← threshold direction needs no model numbers

PHASE 2 — M1 hands off preprocessor → M2 + M3 run in parallel
─────────────────────────────────────────────────────────────────────
M2  ░░░ Run 5-fold CV on 3 models → declare Champion ████████████
M3  ░░░ Run 4 ablations on Champion model             ████████████
M4  ░░░ Extract failure cases from validation set     █████████

PHASE 3 — Final integration (~30 min)
─────────────────────────────────────────────────────────────────────
M4: run final_model.pkl ONCE on test.csv → record test metrics
M5: collect all numbers from M2/M3/M4 → export 12-slide PDF
```

---

## Member 1 — Data Engineer & Pipeline Lead

**Assignment Part:** A  
**Slides owned:** 3 (Data Engineering), 4 (Pipeline Architecture)  
**Blocking for:** Everyone else — complete this first

### Phase 1 (independent, start immediately)
- EDA: churn rate by feature, distribution plots, correlation heatmap
- Feature engineering:
  - `ChargesPerMonth = TotalCharges / tenure`
  - Collapse "No internet service" → binary `HasInternet` flag
  - `tenure_bin` buckets (e.g. 0–12, 13–36, 37–72 months)
- Build sklearn `Pipeline` using `ColumnTransformer`:
  - Numerical features → `RobustScaler`
  - Binary categoricals → `OrdinalEncoder`
  - Multi-class categoricals → `OneHotEncoder`
  - Class imbalance → `SMOTE` via `imblearn.pipeline.Pipeline` (only inside CV folds — never leaks)
- Stratified 80/20 train/validation split → export `X_train, X_val, y_train, y_val`

**Hand-off artifact:** `preprocessor` pipeline object for M2 and M3 to import

### Slides
- **Slide 3:** Feature inventory table, engineered features with rationale, class imbalance stats (22.5% churn)
- **Slide 4:** Pipeline diagram showing `ColumnTransformer` steps in order, where SMOTE sits, why each step prevents data leakage

---

## Member 2 — Model Selection Lead

**Assignment Part:** B  
**Slides owned:** 5 (Model Comparison), 6 (Champion Justification)  
**Blocking for:** M3 (champion identity), M4 (validation predictions)

### Phase 1 (independent — no pipeline needed yet)
- Define 3 classifiers with default params:
  1. `LogisticRegression(max_iter=1000)` — linear family
  2. `RandomForestClassifier(random_state=42)` — tree ensemble
  3. `GradientBoostingClassifier(random_state=42)` — boosting ensemble
- Write the CV loop skeleton: `StratifiedKFold(n_splits=5)`, metric collection (ROC-AUC + F1)
- Prepare results table template (columns: Model | ROC-AUC Mean | ROC-AUC Std | F1 Mean | F1 Std)

### Phase 2 (after M1 hand-off)
- Wrap each classifier with M1's preprocessor in a full `Pipeline`
- Run 5-fold CV; record **ROC-AUC mean ± std** (primary) and F1 mean ± std for each model
- Declare **Champion** = best mean + lowest std
- Generate `val_predictions` and `val_probabilities` on the validation set → share with M4

**Hand-off artifact:** `val_probabilities` array + declared Champion classifier type

### Slides
- **Slide 5:** CV comparison table (all 3 models, both metrics)
- **Slide 6:** Champion justification — why this model family suits churn prediction on this dataset

---

## Member 3 — Ablation & Tuning Lead

**Assignment Part:** C  
**Slides owned:** 7 (Ablation Log), 8 (Final Stability Metrics)  
**Depends on:** M2 declaring the Champion

### Phase 1 (independent — prepare before M2 finishes)
- Write a parameterised `run_ablation(pipeline_config)` function that returns CV mean ± std
- Draft 4 ablation hypotheses (fill in results later):
  1. Add `tenure_bin` engineered feature
  2. Apply SMOTE class balancing
  3. Hyperparameter search: `n_estimators` + `max_depth` (≤50 `RandomizedSearchCV` iterations)
  4. Set `class_weight='balanced'`
- Prepare ablation log table template: `Hypothesis | Controlled Change | CV Mean ± Std | Conclusion`

### Phase 2 (after M2 picks Champion — can run alongside M2's CV)
- Swap in Champion classifier; run all 4 ablations sequentially
- Fill ablation log with actual numbers and conclusions
- Re-run final CV on best ablation config → this becomes the **final model**
- Save `final_model.pkl` → hand to M4 for test set evaluation

**Hand-off artifact:** `final_model.pkl`

### Slides
- **Slide 7:** Filled ablation log table (4 rows)
- **Slide 8:** Final CV Mean ± Std vs pre-ablation baseline (delta column)

---

## Member 4 — Failure Analysis Lead

**Assignment Part:** D  
**Slides owned:** 9 (Failure Cases), 10 (Proposed Fixes)  
**Depends on:** M2 for `val_probabilities`; M3 for `final_model.pkl` (test eval only)

### Phase 1 (independent — explore while waiting)
- Explore training data: identify feature combinations correlated with churn that might confuse the model
- Write failure extraction function:
  ```python
  # Filter: wrong predictions where model confidence > 0.75
  # Return: feature rows + true label + predicted label + probability
  ```
- Prepare analysis table template (10 rows: 5 FP + 5 FN)

### Phase 2 (after M2 delivers `val_probabilities`)
- Extract exactly **5 False Positives** + **5 False Negatives** (confidence > 0.75)
- For each row: write 2–3 sentences mechanically explaining the failure based on feature values
- Group into failure patterns; propose one specific technical fix per pattern (e.g., "add `Contract × MonthlyCharges` interaction feature")

### Phase 3 (last step — after M3 finalises `final_model.pkl`)
- Run `final_model.pkl` **exactly once** on `test.csv` → record ROC-AUC and F1
- Pass final test metrics to M5 for slides 2 and 8

### Slides
- **Slide 9:** Failure case table — 10 rows with key feature values highlighted, true vs predicted label
- **Slide 10:** Root causes grouped by failure type + proposed technical fixes

---

## Member 5 — Business Analyst & Report Lead

**Assignment Part:** E + overall report compilation  
**Slides owned:** 1, 2, 11, 12  
**Mostly independent — can write most content before model results arrive**

### Phase 1 (fully independent — start immediately)
- Threshold direction analysis (qualitative, needs no model numbers):
  - False Negative (missed churner) = lost customer = higher cost
  - False Positive (unnecessary retention offer) = small cost
  - Conclusion: lower threshold below 0.5 → more aggressive at flagging potential churners
- Build full 12-slide deck skeleton with section headers for all members' slides
- Draft Executive Summary structure: Problem → Approach → Result [placeholder] → Recommendation
- Write Slide 11 narrative in full (threshold reasoning is qualitative)

### Phase 2 (fill in numbers as they arrive)
- Insert CV comparison table from M2 → Slide 5
- Insert ablation log from M3 → Slides 7/8
- Insert failure cases from M4 → Slides 9/10
- Insert final test metrics from M4 → Slides 2 and 8
- Export final 12-slide PDF

### Slides
- **Slide 1:** Title, all team members + roles, GitHub repo link
- **Slide 2:** Executive summary (problem → approach → key metric → recommendation)
- **Slide 11:** Business decision — which error is costlier, threshold direction with justification
- **Slide 12:** Individual contributions table (one row per member, exact tasks completed)

---

## Shared Rules

| Rule | Detail |
|------|--------|
| Single notebook | Add your section under a clear `## Part X — Member Name` header |
| Data split | M1 creates it once; everyone else imports `X_train, X_val, y_train, y_val` |
| Test set | Touched **exactly once** by M4 in Phase 3, after all tuning is done |
| No leakage | All `fit()` calls must be inside the Pipeline — never on raw train data directly |
| Ablation limit | Exactly 4 experiments (Part C hard constraint) |
| Hyperparameter search | ≤50 total iterations (Part C hard constraint) |
| Failure cases | Exactly 10 rows: 5 FP + 5 FN, all with confidence > 0.75 |

---

## Final Verification Checklist

- [ ] Notebook runs top-to-bottom without errors
- [ ] All scalers/encoders/SMOTE are inside Pipeline steps (no external `.fit()`)
- [ ] Test set prediction cell exists only once, at the end
- [ ] Ablation table has exactly 4 rows
- [ ] Failure table has exactly 10 rows (5 FP + 5 FN)
- [ ] PDF has exactly 12 slides
- [ ] Slide 12 names each member with their specific tasks
