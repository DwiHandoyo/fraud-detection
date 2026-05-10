# data-validation

Great Expectations suite untuk dataset dan request inference. Memenuhi
spec IF5251 nomor 3a (validasi data otomatis).

## Apa yang divalidasi

| Suite | Target | Kapan dijalankan |
|-------|--------|------------------|
| `identity_dataset` | `identity.parquet` (144k rows × 33 fitur) | Batch — sebelum training / re-materialize |
| `identity_request` | Single inference payload | Per request — middleware di ml-service |

### Contoh expectation

| Kolom | Expectation | Alasan |
|-------|-------------|--------|
| `transaction_id` | not null, unique, positive | Entity key |
| `id_11` | between 0 and 100 | rating percentage |
| `id_14` | between -720 and 840 | timezone offset (-12h to +14h) |
| `id_32` | in {16, 24, 32} | screen color depth (bits) |
| `DeviceType` | in {mobile, desktop} | enum |
| `id_15` | in {Found, New, Unknown} | enum |
| `id_30` | not null mostly 50% | sanity (catch all-null drift) |

## Struktur

```
data-validation/
├── suites/
│   ├── identity_dataset.py     # build() → ExpectationSuite (batch)
│   └── identity_request.py     # build() → ExpectationSuite (request)
├── validate_dataset.py         # CLI batch validator + HTML report
├── validate_request.py         # RequestValidator class + lazy singleton
├── reports/                    # generated artifacts
│   ├── dataset_validation.json
│   └── data_docs/index.html
├── gx/                         # GX project state (auto-generated)
├── requirements.txt
└── README.md
```

## Quickstart

### Batch validation

```bash
cd data-validation
../playground/.venv/bin/python validate_dataset.py
```

Validates `feature-store/feature_repo/data/identity.parquet` and writes:
- `reports/dataset_validation.json` — machine-readable summary
- `reports/data_docs/index.html` — Data Docs HTML site

Last run: **40/40 expectations pass on 144 233 rows.**

### Request validation (in-process)

```python
from validate_request import get_validator

v = get_validator()
ok, errors = v.validate({"TransactionAmt": -50})
# ok = False
# errors = [{"expectation": "expect_column_values_to_be_between",
#            "column": "TransactionAmt", "observed": [-50], ...}]
```

### Request validation (via ml-service middleware)

Set `validation.enabled: true` in `ml-service/config.yaml`. Then any invalid
payload to `/predict` or `/explain` returns HTTP 422:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"TransactionAmt": -50}'
# {"detail":{"validation_errors":[...]}}, status 422
```

## Adding new expectations

Edit `suites/identity_dataset.py` or `suites/identity_request.py` and append:

```python
suite.add_expectation(gxe.ExpectColumnValuesToBeBetween(
    column="new_field", min_value=0, max_value=100,
))
```

Re-run `validate_dataset.py` to regenerate the HTML report.

## Limitasi

- Suite saat ini hanya identity-side (33 fitur). Saat
  `train_transaction.csv` tersedia, tambahkan suite `transaction_dataset` dan
  `transaction_request` (TransactionAmt, ProductCD, V*, C*, D*, M*).
- Batch validator memuat full parquet ke memory (~700 MB di Pandas). Untuk
  data lebih besar, switch ke `data_sources.add_pandas_filesystem` atau
  Spark backend.
- Request validator membuat ephemeral GX context per process — startup ~500 ms.
  Tidak masalah untuk service longrunning, tapi tidak cocok untuk Lambda /
  fungsi serverless.
