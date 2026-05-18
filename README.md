# Fraud Detection — IF5251 Group Project

Real-Time Fraud Detection system untuk mata kuliah **Engineering AI-Enabled
Systems (IF5251)** di ITB. Dataset: IEEE-CIS Fraud Detection (Vesta + Kaggle).

## Komponen

| # | Folder | Tujuan | Spec |
|---|--------|--------|------|
| 1 | [ml-service/](ml-service/) | FastAPI inference + explanation, adapter pattern model-agnostic | Arch 2b, RAI 5b |
| 2 | [feature-store/](feature-store/) | Feast feature store (online SQLite + offline parquet) | Arch 2a |
| 3 | [data-validation/](data-validation/) | Great Expectations suite (dataset + request) | QA 3a |
| 4 | [ui/](ui/) | Streamlit human review UI | Arch 2c |
| 5 | [monitoring/](monitoring/) | Evidently drift dashboard | Ops 4b |
| 6 | [fairness/](fairness/) | FairLearn audit (DeviceType, OS, Browser) | RAI 5a |
| 7 | [qa-tests/](qa-tests/) | Adversarial + locust load test | QA 3b, 3c |
| 8 | [docs/FMEA.md](docs/FMEA.md) | Failure Mode and Effects Analysis | Req 1c |
| 9 | [.github/workflows/](.github/workflows/) | CI/CD (lint + test + build) | Ops 4a |
| 10 | [playground/](playground/) | EDA notebooks, EBM training, original LGBM | (research) |

## Inventory — di mana model dan data berada

### Model artifacts

| Artifact | Path source (hasil training) | Path deploy (dipakai service) | Ukuran |
|----------|------------------------------|------------------------------|--------|
| LGBM predictor | `playground/lgbm_tuning.pkl` | `ml-service/models/lgbm.pkl` | ~35 MB |
| EBM explainer | `playground/ebm/ebm_model.pkl` | `ml-service/models/ebm.pkl` | ~780 KB |
| Preprocessor | `playground/preprocessing/preprocessor.pkl` | `ml-service/models/preprocessor.pkl` | ~45 KB |

Service load lokasi mana → ditentukan di
[ml-service/config.yaml](ml-service/config.yaml). Update artifact = copy + edit
config + restart (lihat [Deploy artifact baru](#deploy-artifact-baru-ke-ml-service)).

### Datasets

| File | Path | Ukuran | Status di repo | Dari mana |
|------|------|--------|---------------|-----------|
| `train_identity.csv` | `playground/train_identity.csv` | ~25 MB | ✅ committed | [Kaggle IEEE-CIS](https://www.kaggle.com/c/ieee-fraud-detection/data) |
| `identity.parquet` (Feast offline) | `feature-store/feature_repo/data/identity.parquet` | ~5.7 MB | ✅ committed | Regen dari `train_identity.csv` via [seed_data.py](feature-store/seed_data.py) |
| `online_store.db` (Feast online) | `feature-store/feature_repo/data/online_store.db` | ~965 MB | ❌ gitignored (>GitHub limit) | Regen via `./materialize.sh` |
| `registry.db` (Feast metadata) | `feature-store/feature_repo/data/registry.db` | 3 KB | ❌ gitignored | Regen via `./apply.sh` |
| `train_transaction.csv` | (tidak ada di repo) | ~470 MB | ❌ download manual | [Kaggle IEEE-CIS](https://www.kaggle.com/c/ieee-fraud-detection/data) — wajib untuk retrain LGBM |
| `sample_10rows.csv` | `playground/sample_10rows.csv` | 2 KB | ✅ committed | Sample untuk smoke test |
| Identity reports (HTML) | `data-validation/reports/`, `monitoring/reports/`, `fairness/reports/` | varies | ❌ gitignored | Regen via masing-masing `*.py` script |

### Try-it-out walkthrough (5 menit)

Untuk reviewer/dosen yang mau langsung cek sistemnya jalan tanpa baca semua section:

```bash
# 1. Clone + masuk folder
git clone <repo-url> && cd fraud-detection

# 2. Pre-flight (regen Feast online_store.db — wajib, ~30 detik)
cd feature-store
python seed_data.py
./apply.sh
./materialize.sh
cd ..

# 3. Start service
docker compose up -d --build         # build pertama ~5 menit

# 4. Test prediction (lookup features dari Feast)
curl -X POST http://localhost:8000/predict_by_id \
  -H "Content-Type: application/json" \
  -d '{"transaction_id": 2987004}'
# Expected: {"fraud_proba": 0.42, "predicted_label": 0, "model": "lgbm", ...}

# 5. Buka UI di browser
open http://localhost:8501

# 6. Coba reproduce training (EBM saja, tidak butuh train_transaction.csv)
cd playground/ebm
../.venv/bin/python train_ebm.py
# Output: ebm_model.pkl + metrics.json (~1 menit)
```

Detail lengkap untuk training/deploy/monitoring di section
[Training](#training) dan [Quickstart](#quickstart).

## Quickstart

### 1. Setup data + feature store (sekali saja)

```bash
cd playground
python -m venv .venv
source .venv/bin/activate
pip install -r ../ml-service/requirements.txt -r ../feature-store/requirements.txt -r ../data-validation/requirements.txt -r ../ui/requirements.txt -r ../monitoring/requirements.txt -r ../fairness/requirements.txt -r ../qa-tests/requirements.txt

# Seed feature store
cd ../feature-store
python seed_data.py
./apply.sh
./materialize.sh
```

### 2. Run service + UI (Docker)

```bash
docker compose up -d
open http://localhost:8501   # Streamlit UI
open http://localhost:8000/docs  # FastAPI docs
```

### 3. Generate reports (offline)

```bash
# Data validation HTML
cd data-validation && python validate_dataset.py

# Drift report
cd ../monitoring && python generate_report.py

# Fairness audit
cd ../fairness && python audit.py

# Load test (ml-service must be running)
cd ../qa-tests && locust -f load_test.py --host http://localhost:8000 \
    --users 50 --spawn-rate 10 --run-time 60s --headless \
    --html reports/load_test_report.html
```

## Training

### Big picture — Feast di training vs production

Feast adalah **satu sumber kebenaran** untuk fitur. Definisi `FeatureView` di
[feature-store/feature_repo/features.py](feature-store/feature_repo/features.py)
dipakai oleh **dua jalur** yang berbeda:

```
                        FeatureView (features.py)
                                │
            ┌───────────────────┴────────────────────┐
            ▼                                        ▼
   ┌──────────────────┐                    ┌──────────────────┐
   │ OFFLINE store    │                    │ ONLINE store     │
   │ (parquet file)   │                    │ (SQLite)         │
   │ — historis penuh │                    │ — latest per     │
   │ — point-in-time  │                    │   transaction_id │
   └────────┬─────────┘                    └────────┬─────────┘
            │                                       │
            ▼                                       ▼
  get_historical_features()              get_online_features()
            │                                       │
            ▼                                       ▼
  ┌──────────────────┐                    ┌──────────────────┐
  │ TRAINING         │                    │ PRODUCTION       │
  │ train_*.py       │                    │ /predict_by_id   │
  │ (offline batch)  │                    │ (online <10ms)   │
  └──────────────────┘                    └──────────────────┘
```

Definisi fitur sama → **tidak ada training/serving skew**. Memenuhi spec
**Arch 2a**.

### Training scripts

Ada **3 artifact** yang dilatih, di-ranking dari yang paling sering diretrain:

| Artifact | Script training | Butuh data lengkap? | Output |
|----------|----------------|---------------------|--------|
| **EBM** (explainer) | [`playground/ebm/train_ebm.py`](playground/ebm/train_ebm.py) | ❌ (mode distillation) | `ebm_model.pkl` |
| **LGBM** (predictor) | [`playground/2025-05-01_Model.ipynb`](playground/2025-05-01_Model.ipynb) | ✅ butuh `train_transaction.csv` | `lgbm_tuning.pkl` |
| **Preprocessor** | [`playground/preprocessing/fit_preprocessor.py`](playground/preprocessing/fit_preprocessor.py) | ❌ (identity-only) / ✅ (full) | `preprocessor.pkl` |

### Training pipeline (cara pakai Feast saat training)

Best practice: training script **harus baca fitur dari Feast offline store**,
bukan langsung dari CSV. Ini memastikan model dilatih dengan persis fitur yang
nanti dipakai saat serving.

Pseudocode untuk training script (yang akan kompatibel dengan Feast offline):

```python
from feast import FeatureStore
import pandas as pd

store = FeatureStore(repo_path="feature-store/feature_repo")

# Entity dataframe — minimal kolom: entity key + event timestamp
entity_df = pd.read_csv("entity_with_labels.csv")
# kolom: transaction_id, event_timestamp, isFraud

# Tarik fitur historis dari offline store
training_df = store.get_historical_features(
    entity_df=entity_df,
    features=store.get_feature_service("identity_service"),
).to_df()
# training_df sekarang punya: transaction_id, event_timestamp, isFraud,
#   id_01, id_02, ..., DeviceType, DeviceInfo (33 kolom fitur)

# Train model
y = training_df["isFraud"]
X = training_df.drop(columns=["transaction_id", "event_timestamp", "isFraud"])
model.fit(X, y)
```

**Status saat ini:**
- ✅ Feast offline store **sudah ready** — parquet ada di
  `feature-store/feature_repo/data/identity.parquet`
- ⚠️ `playground/2025-05-01_Model.ipynb` (LGBM training) saat ini baca **langsung dari CSV**,
  belum migrasi ke `get_historical_features()`
- ⚠️ `playground/ebm/train_ebm.py` (EBM training) juga baca langsung dari CSV
- 🟡 **Refactor opportunity**: ubah kedua training jalur supaya pakai
  `get_historical_features()`. Demonstrasi konsistensi training/serving yang
  lebih meyakinkan untuk laporan.

### Production pipeline (cara pakai Feast saat serving)

Sudah terimplementasi di
[ml-service/app/feature_client.py](ml-service/app/feature_client.py):

```python
# Dipanggil oleh /predict_by_id dan /explain_by_id
features_df = feast_client.get_features(transaction_id)  # ~5-10ms latency
prediction = predictor.predict_proba(features_df)
```

Online store di-populate via `feast materialize-incremental` (lihat
[feature-store/materialize.sh](feature-store/materialize.sh)).

### Cara training (commands)

#### EBM explainer

```bash
cd playground/ebm
../.venv/bin/python train_ebm.py                              # distillation default
../.venv/bin/python train_ebm.py --top-n 50                   # pakai top-50 fitur
../.venv/bin/python train_ebm.py \
  --transaction-csv /path/to/train_transaction.csv            # supervised mode
```

Output: `playground/ebm/ebm_model.pkl` + `metrics.json`.
Detail: [playground/ebm/README.md](playground/ebm/README.md).

#### Preprocessor

```bash
cd playground/preprocessing
../.venv/bin/python fit_preprocessor.py                       # identity-only
../.venv/bin/python fit_preprocessor.py \
  --full /path/to/train_transaction.csv                       # full features
```

Output: `preprocessor.pkl` (~45 KB).
Detail: [playground/preprocessing/README.md](playground/preprocessing/README.md).

#### LGBM predictor

Notebook: [playground/2025-05-01_Model.ipynb](playground/2025-05-01_Model.ipynb).
Butuh `train_transaction.csv` (~470 MB dari
[Kaggle IEEE-CIS](https://www.kaggle.com/c/ieee-fraud-detection/data)).

Saat ini belum ada `train_lgbm.py` standalone — tinggal extract dari notebook
kalau perlu CI/CD continuous training.

### Deploy artifact baru ke ml-service

```bash
# 1. Copy .pkl yang baru dilatih
cp playground/ebm/ebm_model.pkl ml-service/models/ebm.pkl
cp playground/lgbm_tuning.pkl ml-service/models/lgbm.pkl
cp playground/preprocessing/preprocessor.pkl ml-service/models/preprocessor.pkl

# 2. Bump version di config.yaml untuk audit trail
#    Edit ml-service/config.yaml: predictor.version: "1.1"

# 3. Re-materialize feature store kalau training data berubah
cd feature-store && ./materialize.sh

# 4. Rebuild + restart ml-service
cd .. && docker compose up -d --build

# 5. Verify version baru
curl http://localhost:8000/health
# {"predictor": "lgbm/1.1", ...}
```

## Arsitektur

```
                 ┌─────────────────┐
                 │  Streamlit UI   │  http://localhost:8501
                 │  (ui/)          │
                 └────────┬────────┘
                          │ HTTP
                          ▼
┌──────────────┐   ┌─────────────────┐   ┌──────────────────┐
│ Great        │◄──│  ml-service     │──►│ Feast online     │
│ Expectations │   │  (FastAPI)      │   │ store (SQLite)   │
│ validator    │   │                 │   │                  │
│              │   │  ┌───────────┐  │   └────────┬─────────┘
│ data-        │   │  │ LGBM      │  │            │
│ validation/  │   │  │ adapter   │  │            ▼
└──────────────┘   │  │ (predict) │  │   ┌──────────────────┐
                   │  └───────────┘  │   │ Feast offline    │
                   │  ┌───────────┐  │   │ (parquet)        │
                   │  │ EBM       │  │   │ feature-store/   │
                   │  │ adapter   │  │   └──────────────────┘
                   │  │ (explain) │  │
                   │  └───────────┘  │
                   └─────────────────┘
                          │
                          ▼
                   ┌──────────────────┐
                   │ Predictions log  │
                   │ (stdout JSON)    │
                   └─────────┬────────┘
                             │ batched
                             ▼
                   ┌──────────────────┐    ┌──────────────────┐
                   │ Evidently        │    │ FairLearn audit  │
                   │ (monitoring/)    │    │ (fairness/)      │
                   └──────────────────┘    └──────────────────┘
```

## Spec coverage

| Spec | Status | Komponen |
|------|--------|----------|
| Req 1c — FMEA | ✅ | [docs/FMEA.md](docs/FMEA.md) |
| Arch 2a — Feature Store (Feast) | ✅ | [feature-store/](feature-store/) |
| Arch 2b — Microservice + container | ✅ | [ml-service/](ml-service/) |
| Arch 2c — Human review UI | ✅ | [ui/](ui/) |
| QA 3a — Data validation | ✅ | [data-validation/](data-validation/) |
| QA 3b — Adversarial test | ✅ | [qa-tests/adversarial.py](qa-tests/adversarial.py) |
| QA 3c — Load test | ✅ | [qa-tests/load_test.py](qa-tests/load_test.py) |
| Ops 4a — CI/CD | ✅ | [.github/workflows/ci.yml](.github/workflows/ci.yml) |
| Ops 4b — Monitoring (Evidently) | ✅ | [monitoring/](monitoring/) |
| RAI 5a — Fairness audit | ✅ | [fairness/](fairness/) |
| RAI 5b — Explainability | ✅ | [ml-service/app/adapters.py](ml-service/app/adapters.py) `EBMAdapter.explain` |
| RAI 5c — Encryption + model inversion | ⚠️ partial | dijelaskan di FMEA, mitigasi belum diimplementasi |

## Model architecture

**Predictor:** LightGBM (424 fitur, di-train dari notebook
[playground/2025-05-01_Model.ipynb](playground/2025-05-01_Model.ipynb)).

**Explainer:** EBM (Explainable Boosting Machine) sebagai distillation
surrogate. Detail:
[playground/ebm/](playground/ebm/), training: `python playground/ebm/train_ebm.py`.

**Swap model:** edit [ml-service/config.yaml](ml-service/config.yaml) baris
`predictor.type` + `path`. Tambah model baru = ~30 baris adapter class +
1 entry di `ADAPTERS` dict. Lihat [ml-service/README.md](ml-service/README.md).

## Limitasi yang didokumentasikan

`train_transaction.csv` (~470 MB) tidak ada di repo karena ukurannya. Akibat:
- LGBM trained pada notebook saat dataset masih ada — pkl tetap valid
- EBM dilatih sebagai distillation (LGBM-as-teacher) pada train_identity.csv
- Feature store hanya berisi 33 fitur identity (bukan 424 fitur penuh)
- Fairness audit pakai LGBM prediction sebagai proxy label (bukan ground truth)
- Evidently drift report split static dataset (bukan reference vs production logs)

Saat dataset penuh tersedia, semua komponen di atas tinggal dijalankan ulang —
tidak ada perubahan kode yang diperlukan.

## Server target (info dari spec)

```
CPU:     2× Intel Xeon Silver 4114 (20 core, 2.20 GHz)
RAM:     192 GB DDR4
Storage: 1 TB HDD + 480 GB SSD
GPU:     NVIDIA P100 12 GB
Network: Mellanox 25 Gb NIC
```

Server ephemeral — semua artifact bisa di-regenerate dari script + data sumber.

## Tim & kontribusi

(Diisi oleh tim — placeholder)
