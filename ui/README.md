# ui — Streamlit multi-page console

Memenuhi spec IF5251 nomor **Architecture 2c** (human validation) +
**RAI 5b** (display explainability).

## Pages

| File | Tujuan |
|------|--------|
| [`app.py`](app.py) | Landing — overview + service health banner |
| [`pages/1_Score_by_ID.py`](pages/1_Score_by_ID.py) | Reviewer mode — lookup by transaction_id, gunakan Feast online store |
| [`pages/2_What_if_Scoring.py`](pages/2_What_if_Scoring.py) | Audit mode — input fitur manual untuk skenario hipotetis |
| [`pages/3_Audit_Log.py`](pages/3_Audit_Log.py) | History keputusan reviewer dengan filter + search + CSV export |
| [`pages/4_Service_Status.py`](pages/4_Service_Status.py) | Health, model versions, OpenAPI docs links |

Streamlit otomatis bikin sidebar navigation dari folder `pages/` — urut sesuai
prefix angka (`1_*`, `2_*`, dst).

## Shared module

[`api_client.py`](api_client.py) — single source untuk:
- HTTP client (predict_by_id, explain_by_id, predict, explain, health)
- Audit log read/write (`decisions.jsonl`)
- Bar chart renderer untuk explanation
- Prediction header metrics renderer

Semua pages import dari sini → tidak ada duplikasi logic.

## Run

### Local
```bash
cd ui
../playground/.venv/bin/python -m pip install -r requirements.txt
../playground/.venv/bin/streamlit run app.py
# Buka http://localhost:8501
```

### Docker
```bash
docker build -t fraud-ui .
docker run -p 8501:8501 -e ML_SERVICE_URL=http://host.docker.internal:8000 fraud-ui
```

### Compose (recommended)
```bash
docker compose up -d   # dari root fraud-detection/
```

## Env vars

| Var | Default | Fungsi |
|-----|---------|--------|
| `ML_SERVICE_URL` | `http://localhost:8000` | URL ml-service |
| `DECISIONS_LOG` | `decisions.jsonl` | Path file audit |

## Audit log format

`decisions.jsonl` — satu JSON object per baris:
```json
{
  "ts": "2026-05-10T15:30:00Z",
  "transaction_id": 2987004,
  "source": "by_id",
  "decision": "approve_legit",
  "model_proba": 0.42,
  "model_label": 0,
  "model": "lgbm",
  "note": ""
}
```

Field `source`: `by_id` (dari Score by ID page) atau `whatif` (dari What-if Scoring page).
Field `transaction_id`: `null` kalau `source = whatif` (tidak ada ID untuk skenario hipotetis).

## Adding a new page

1. Buat file `pages/N_Page_Name.py` (N = nomor urut, name di-underscored)
2. Import dari `api_client` kalau perlu HTTP / log
3. Streamlit otomatis pickup, restart tidak perlu untuk file baru

Contoh skeleton:
```python
import streamlit as st
from api_client import health

st.set_page_config(page_title="My Page", layout="wide")
st.title("My Page")
# ... isi page ...
```
