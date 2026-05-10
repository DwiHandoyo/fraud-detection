# ui — Streamlit human review

Memenuhi spec IF5251 nomor **Architecture 2c**: antarmuka untuk manusia
memvalidasi prediksi fraud dengan confidence score rendah.

## Apa yang dilakukan

1. Reviewer input `transaction_id`
2. UI panggil `POST /predict_by_id` di ml-service → tampilkan
   probability + label + flag "LOW CONFIDENCE" kalau proba ∈ [0.30, 0.70]
3. UI panggil `POST /explain_by_id` → render bar chart kontribusi 15 fitur
   teratas (warna merah = fraud-pushing, hijau = protective)
4. Reviewer klik **Approve** / **Confirm fraud** / **Need more info**
5. Keputusan di-append ke `decisions.jsonl` (audit trail)

## Run

```bash
# Prereq: ml-service jalan di http://localhost:8000
cd ui
../playground/.venv/bin/python -m pip install -r requirements.txt
../playground/.venv/bin/streamlit run app.py
# Buka http://localhost:8501
```

Atau via Docker:
```bash
docker build -t fraud-ui .
docker run -p 8501:8501 -e ML_SERVICE_URL=http://host.docker.internal:8000 fraud-ui
```

## Sample TransactionIDs

Dari `train_identity.csv`: `2987004`, `2987008`, `2987010`, `2987011`,
`2987016`, `2987017`, `2987022`, `2987038`.

## Audit log format

`decisions.jsonl` — satu JSON per baris:
```json
{"ts":"2026-05-10T15:30:00Z","transaction_id":2987004,
 "decision":"approve_legit","model_proba":0.42,"model_label":0,
 "model":"lgbm","note":""}
```

## Env vars

| Var | Default | Fungsi |
|-----|---------|--------|
| `ML_SERVICE_URL` | `http://localhost:8000` | URL ml-service |
| `DECISIONS_LOG` | `decisions.jsonl` | Path file audit |
