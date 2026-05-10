# ml-service

Model-agnostic fraud detection inference service. FastAPI app that loads
predictor and explainer adapters from `config.yaml`. Swap models by editing
the config — no code change required.

## Architecture

```
app/
├── main.py        FastAPI app, 3 endpoints (~70 lines)
├── adapters.py    Predictor/Explainer protocol + LGBM + EBM (~120 lines)
├── schemas.py     Pydantic request/response (~80 lines)
├── settings.py    config.yaml loader (~25 lines)
└── preprocess.py  Imported because preprocessor.pkl needs the class

models/            Model artifacts (.pkl)
config.yaml        Single source of model selection
tests/             pytest unit + integration tests
```

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Service status + which models are loaded |
| POST | `/predict` | Returns fraud probability + label (caller passes features) |
| POST | `/explain` | Returns per-feature contributions (caller passes features) |
| POST | `/predict_by_id` | Lookup features from feature store, then predict |
| POST | `/explain_by_id` | Lookup features from feature store, then explain |
| GET | `/docs` | OpenAPI dashboard (auto-generated) |

The `_by_id` variants require a running feature store. See
[../feature-store/README.md](../feature-store/README.md) for setup.

## Run locally (no Docker)

```bash
cd ml-service
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open http://localhost:8000/docs for the interactive UI.

## Run with Docker

```bash
docker build -t fraud-ml .
docker run -p 8000:8000 fraud-ml
```

## Quick test

```bash
curl -s http://localhost:8000/health | jq

curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"TransactionAmt": 250, "card4": "visa", "P_emaildomain": "gmail.com"}' | jq

curl -s -X POST http://localhost:8000/explain \
  -H "Content-Type: application/json" \
  -d '{"TransactionAmt": 250, "card4": "visa", "id_31": "chrome 62.0"}' | jq

# With feature store: caller only needs transaction_id
curl -s -X POST http://localhost:8000/predict_by_id \
  -H "Content-Type: application/json" \
  -d '{"transaction_id": 2987004}' | jq

curl -s -X POST http://localhost:8000/explain_by_id \
  -H "Content-Type: application/json" \
  -d '{"transaction_id": 2987004}' | jq
```

## Swapping models

Open `config.yaml`, change one line:

```yaml
predictor:
  type: ebm           # was: lgbm
  path: models/ebm.pkl
```

Restart the service. Done.

## Adding a new model type

1. Add a class to `adapters.py` matching the `Predictor` (and optionally
   `Explainer`) Protocol — needs `name`, `version`, `feature_names`,
   `predict_proba(X)`.
2. Register in the `ADAPTERS` dict at the bottom of `adapters.py`.
3. Reference `type: <your_name>` in `config.yaml`.

Example skeleton (~30 lines):

```python
class XGBoostAdapter:
    def __init__(self, path, version="1.0"):
        self.model = joblib.load(path)
        self.feature_names = self.model.feature_names_in_.tolist()
        self.name = "xgboost"
        self.version = version
    def predict_proba(self, X):
        X_aligned = _align(X, self.feature_names)
        X_aligned = _factorize_objects(X_aligned)
        return self.model.predict_proba(X_aligned)[:, 1].tolist()

ADAPTERS["xgboost"] = XGBoostAdapter
```

## Testing

```bash
pytest tests/
```

## Notes / known limitations

- Current LGBM adapter uses on-the-fly `pd.factorize` for unknown categoricals
  (the original LabelEncoder from training was not persisted). For production-
  grade consistency, fit a full preprocessor on `train_transaction.csv` +
  `train_identity.csv` and wire it through.
- EBM adapter is the surrogate distillation model from `playground/ebm/`. It
  uses 25 identity features only.
