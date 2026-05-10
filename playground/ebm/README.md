# EBM (Explainable Boosting Machine) Demo

Glass-box alternative to the LightGBM fraud model — same gradient boosting math,
but with **per-feature shape functions you can plot and audit**.

## Mode

This demo runs in **knowledge distillation mode**:

- Teacher: existing `lgbm_tuning.pkl` (425 features, black-box)
- Student: EBM trained on a subset of features, using LGBM probabilities as soft labels
- Result: a transparent surrogate model that mimics the teacher's behavior on the
  identity feature space

Why distillation? Only `train_identity.csv` is checked into the repo. The full
`train_transaction.csv` (Kaggle IEEE-CIS data) is not present, so we cannot use
the real `isFraud` label. The teacher–student setup lets us train a working EBM
on the data we have while still demonstrating its interpretability.

If you have `train_transaction.csv`, pass `--transaction-csv path/to/train_transaction.csv`
to `train_ebm.py` and it will switch to **supervised mode** with the real label.

## Quickstart

```bash
# 1. Install (uses the existing playground venv)
cd /Users/mac/Documents/ai-prod/tubes/fraud-detection/playground
.venv/bin/pip install -r ebm/requirements.txt

# 2. Train the EBM (distillation mode, runs out-of-the-box)
.venv/bin/python ebm/train_ebm.py

# 3. Generate explanations (shape functions, local explanations, HTML report)
.venv/bin/python ebm/explain_ebm.py
```

Outputs land in `ebm/`:

- `ebm_model.pkl` — trained EBM
- `metrics.json` — train/val AUC, fidelity vs teacher
- `plots/shape_*.png` — global shape function for each feature
- `plots/local_*.png` — local explanation for sample predictions
- `plots/teacher_vs_student.png` — fidelity scatter plot
- `report.html` — interactive interpret.ml dashboard (open in browser)

## What you get from EBM

Each feature has a **shape function** `f(x)` showing its contribution to log-odds:

```
log-odds(fraud) = intercept + Σ fⱼ(xⱼ) + Σ fᵢⱼ(xᵢ, xⱼ)
```

You can read off, for any input value of a feature, exactly how much it pushes
the prediction toward fraud or not — no SHAP or LIME post-hoc trickery needed.

## Files

| File | Purpose |
|------|---------|
| `train_ebm.py` | Load data, prepare features, train EBM, save model + metrics |
| `explain_ebm.py` | Load trained EBM, generate shape function plots + HTML report |
| `requirements.txt` | Python dependencies |
| `plots/` | Generated visualizations |
