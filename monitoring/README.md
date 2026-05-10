# monitoring — Evidently drift dashboard

Memenuhi spec IF5251 nomor **Operations 4b**: pemantauan model & data drift.

## Apa yang dilakukan

`generate_report.py` membuat laporan HTML yang membandingkan dua snapshot
data:

- **Reference**: 72k baris pertama `identity.parquet`
- **Current**: 72k baris terakhir

Untuk 15 kolom (8 numerik + 7 kategorikal), Evidently menghitung:

- Distribusi reference vs current
- Drift score (Wasserstein, JS divergence, Kolmogorov-Smirnov)
- Missing-value trend
- Categorical class balance

## Run

```bash
cd monitoring
../playground/.venv/bin/python -m pip install -r requirements.txt
../playground/.venv/bin/python generate_report.py
# Output: reports/drift_report.html
open reports/drift_report.html
```

## Interpretasi output

| File | Isi |
|------|-----|
| `reports/drift_report.html` | Interactive dashboard (5 MB) |
| `reports/drift_summary.json` | Machine-readable, digunakan oleh CI/CD untuk gating |

Last run: **0/15 kolom drifted** (expected — train_identity.csv adalah static
dataset, dipotong jadi dua tidak menghasilkan drift nyata).

## Untuk monitoring real

Saat ml-service mulai produksi prediction log:

1. Tambah structured logging di [ml-service/app/main.py](../ml-service/app/main.py):
   ```python
   logger.info(json.dumps({
       "event": "prediction",
       "transaction_id": req.transaction_id,
       "fraud_proba": proba,
       "model": predictor.name,
       "ts": datetime.utcnow().isoformat() + "Z",
   }))
   ```
2. Pipe stdout ke file: `uvicorn ... 2>&1 | tee logs/predictions.jsonl`
3. Update `generate_report.py` baca dari `logs/predictions.jsonl` untuk
   "current", `train_identity.csv` untuk "reference"
4. Schedule `generate_report.py` di cron / GitHub Actions

## Limitasi

- Evidently 0.7+ API masih dalam evolusi — `Report.run()` return signature
  bisa berubah antar minor version
- HTML 5 MB cukup besar; untuk batch monitoring 100+ snapshot, simpan JSON
  saja dan render HTML on-demand
- Tidak ada model performance drift (butuh `isFraud` ground truth yang tidak
  tersedia)
