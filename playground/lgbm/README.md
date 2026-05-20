# playground/lgbm — LGBM training script

Standalone version of LGBM training previously living in
[`../2025-05-01_Model.ipynb`](../2025-05-01_Model.ipynb) (cell 5 + cell 10).
CI-runnable, reproducible, no Jupyter dependency.

## Why a script, not a notebook?

- Notebooks aren't friendly to CI/CD (cell ordering, hidden state)
- Easier to diff changes in PR
- Can be triggered by GitHub Actions for continuous retraining (spec **Ops 4a**)

## Prereq

```bash
cd playground/lgbm
../.venv/bin/python -m pip install -r requirements.txt
```

You also need `train_transaction.csv` (~470 MB) which is NOT in the repo.
Download from
[Kaggle IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection/data)
and place it anywhere (path is a CLI arg).

## Quick smoke test (~30 detik dengan `--quick`)

```bash
python train_lgbm.py \
  --transaction-csv /path/to/train_transaction.csv \
  --identity-csv ../train_identity.csv \
  --quick
```

`--quick` = `n_estimators=200` tanpa early stopping. Expected val AUC ~0.85.

## Full training (~10–15 menit di mesin biasa)

```bash
python train_lgbm.py \
  --transaction-csv /path/to/train_transaction.csv \
  --identity-csv ../train_identity.csv
```

Expected val AUC: ~0.92 (matches the original notebook).

## Output files

| File | Isi |
|------|-----|
| `lgbm_model.pkl` | fitted `LGBMClassifier`, ~35 MB |
| `label_encoders.pkl` | dict `{column: LabelEncoder}`, ~5 KB. **Penting**: pakai ini di inference untuk consistency dengan training (fix bug `pd.factorize` per-request di ml-service). |
| `metrics.json` | AUC, hyperparams, confusion matrix, classification report |
| `feature_importance.csv` | ranked features by gain + split |

## Pipeline (mirror notebook)

1. Load + merge `train_transaction.csv` + `train_identity.csv` on `TransactionID`
2. Drop high-missing/ID columns: `id_07, id_21–id_27, TransactionID`
3. Drop outliers `TransactionAmt > 20000`
4. Fillna: kategorikal → `'null'`, numerik → `-999`
5. `LabelEncoder` per kolom kategorikal (persisted)
6. Stratified train/val split 80/20
7. `lgb.LGBMClassifier` dengan hyperparams default (cell 10):
   - `learning_rate=0.01`
   - `max_depth=12`
   - `n_estimators=10000`
   - `bagging_fraction=0.8`
   - `feature_fraction=0.4`
   - `boosting_type='gbdt'`
8. Early stopping pada validation AUC (50 rounds)

## CLI options

| Flag | Default | Catatan |
|------|---------|---------|
| `--transaction-csv` | (required) | Path ke train_transaction.csv |
| `--identity-csv` | `../train_identity.csv` | Default ke file di repo |
| `--output-dir` | `.` | Output `.pkl` dan metrics |
| `--learning-rate` | 0.01 | LGBM LR |
| `--max-depth` | 12 | LGBM max depth |
| `--n-estimators` | 10000 | Max trees (early stopping akan berhenti lebih awal) |
| `--test-size` | 0.2 | Val split |
| `--random-state` | 42 | Reproducibility |
| `--early-stopping-rounds` | 50 | Stop kalau val AUC tidak naik N rounds |
| `--quick` | (off) | Smoke mode: n_est=200, no early stopping |

## Deploy ke ml-service

```bash
# Copy artifact baru
cp lgbm_model.pkl ../../ml-service/models/lgbm.pkl
cp label_encoders.pkl ../../ml-service/models/label_encoders.pkl

# Bump version di ml-service/config.yaml
#   predictor.version: "1.1"

# Restart
cd ../.. && docker compose up -d --build ml-service
```

## Catatan untuk ml-service adapter

`label_encoders.pkl` saat ini **belum** dipakai oleh `LGBMAdapter`. Adapter
masih melakukan `pd.factorize()` per-request (lihat
[FMEA LGBM-5](../../docs/FMEA.md)). Refactor adapter untuk load encoders:

```python
class LGBMAdapter:
    def __init__(self, path, version="1.0", encoders_path=None):
        self.model = joblib.load(path)
        self.encoders = joblib.load(encoders_path) if encoders_path else None
        ...
    def predict_proba(self, X):
        X = _align(X, self.feature_names)
        if self.encoders:
            for col, le in self.encoders.items():
                if col in X.columns and X[col].dtype == "object":
                    X[col] = X[col].map(lambda v: le.transform([str(v)])[0]
                                        if str(v) in le.classes_ else 0)
        else:
            X = _factorize_objects(X)  # legacy fallback
        return self.model.predict_proba(X)[:, 1].tolist()
```

Refactor itu di luar scope script ini (artifact-nya saja yang disiapkan).
