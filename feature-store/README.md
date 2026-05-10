# feature-store

Feast feature store untuk fraud detection. Menyediakan lookup fitur per
`transaction_id` ke ml-service, sehingga caller tidak perlu mengirim 33 fitur
mentah di setiap request.

## Struktur

```
feature-store/
├── feature_repo/
│   ├── feature_store.yaml      # config Feast (registry, online/offline store)
│   ├── features.py             # Entity + FeatureView + FeatureService
│   └── data/
│       ├── identity.parquet    # source — dibuat oleh seed_data.py
│       ├── registry.db         # dibuat oleh `feast apply`
│       └── online_store.db     # dibuat oleh `feast materialize`
├── seed_data.py                # CSV → parquet
├── apply.sh                    # wrapper feast apply
├── materialize.sh              # wrapper feast materialize-incremental
├── requirements.txt
└── README.md
```

## Komponen

| Komponen | Pilihan | Alasan |
|----------|---------|--------|
| Online store | SQLite | File-based, restore mudah dari offline saat server crash |
| Offline store | FileSource (parquet) | Dataset lokal, tidak butuh BigQuery/Snowflake |
| Entity | `transaction` (join key `transaction_id`) | Sesuai TransactionID di IEEE-CIS |
| FeatureView | `identity_view` (33 fitur) | Cocok dengan train_identity.csv |
| FeatureService | `identity_service` | Diakses oleh ml-service |

## Quickstart

Asumsi venv di `../playground/.venv` sudah ada `feast[sqlite]`.

```bash
# 1. Seed data — convert train_identity.csv → identity.parquet
../playground/.venv/bin/python seed_data.py

# 2. Apply — register definitions di registry.db
./apply.sh

# 3. Materialize — populate SQLite online store dari parquet
./materialize.sh

# 4. Verify lookup
../playground/.venv/bin/python -c "
from feast import FeatureStore
store = FeatureStore(repo_path='feature_repo')
result = store.get_online_features(
    features=store.get_feature_service('identity_service'),
    entity_rows=[{'transaction_id': 2987004}],
).to_dict()
print(result['id_30'], result['id_31'], result['DeviceType'])
"
# Expected: ['Android 7.0'] ['samsung browser 6.2'] ['mobile']
```

## Integrasi ke ml-service

ml-service membaca `repo_path` dari `config.yaml`:

```yaml
feature_store:
  enabled: true
  repo_path: ../feature-store/feature_repo
  service_name: identity_service
```

Saat `enabled: true`, dua endpoint baru aktif:

```bash
# Predict pakai feature lookup (caller cuma kirim transaction_id)
curl -X POST http://localhost:8000/predict_by_id \
  -H "Content-Type: application/json" \
  -d '{"transaction_id": 2987004}'

# Explain pakai feature lookup
curl -X POST http://localhost:8000/explain_by_id \
  -H "Content-Type: application/json" \
  -d '{"transaction_id": 2987004}'
```

Endpoint lama (`/predict`, `/explain`) tetap berfungsi untuk caller dengan
fitur pre-computed — tidak ada breaking change.

## Re-materialize setelah server crash

Server ephemeral → file di `data/` bisa hilang. Restore:

```bash
./apply.sh                          # rebuild registry.db
./materialize.sh                    # rebuild online_store.db dari parquet
```

`identity.parquet` (5.7 MB) cukup di-backup ke storage eksternal. Online store
(1 GB SQLite) bisa di-regenerate kapan saja dari parquet.

## Demonstrasi training/serving consistency

Bukti bahwa fitur yang dipakai untuk training sama dengan yang dipakai serving:

```python
from feast import FeatureStore
import pandas as pd

store = FeatureStore(repo_path="feature_repo")

# Training: get_historical_features → point-in-time correct
entity_df = pd.DataFrame({
    "transaction_id": [2987004, 2987008],
    "event_timestamp": pd.to_datetime(["2018-06-01", "2018-06-01"]),
})
training_df = store.get_historical_features(
    entity_df=entity_df,
    features=store.get_feature_service("identity_service"),
).to_df()

# Serving: get_online_features → low-latency
serving_df = store.get_online_features(
    features=store.get_feature_service("identity_service"),
    entity_rows=[{"transaction_id": 2987004}],
).to_df()

# Same FeatureView → same definitions → same values
```

## Limitasi yang disengaja

- **Tidak ada C/D historical aggregations.** `train_transaction.csv` tidak
  tersedia di repo (perlu download Kaggle ~470 MB). Saat tersedia, tambahkan
  FeatureView baru `transaction_view` dan FeatureService gabungan.
- **Synthetic event_timestamp.** `train_identity.csv` tidak punya kolom
  timestamp. `seed_data.py` membuat timestamp monotonic dari reference date
  2017-12-01 (konvensi komunitas Kaggle).
- **Tidak ada streaming ingestion.** Re-materialize dijalankan manual
  (`materialize.sh`). Untuk sistem real-time fraud detection sungguhan, pasang
  Kafka/Flink → push-based ingestion ke online store.
- **TTL 10 tahun.** Dataset statis, tidak ada concept of staleness.
