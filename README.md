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

Semua model dapat dilatih ulang dari data sumber. Output disimpan sebagai
`.pkl` yang langsung dipakai oleh [ml-service](ml-service/) lewat
[config.yaml](ml-service/config.yaml).

### EBM explainer (glass-box)

Script standalone — bisa langsung dijalankan, tidak butuh `train_transaction.csv`.

```bash
cd playground/ebm
../.venv/bin/python train_ebm.py                 # mode default: distillation dari LGBM
../.venv/bin/python train_ebm.py --top-n 50      # pakai top-50 fitur (default 25)

# Kalau punya train_transaction.csv (real isFraud labels):
../.venv/bin/python train_ebm.py \
  --transaction-csv /path/to/train_transaction.csv
```

Output:
- `playground/ebm/ebm_model.pkl` — model siap pakai
- `playground/ebm/metrics.json` — AUC, fidelity vs LGBM teacher
- (opsional) `playground/ebm/explain_ebm.py` — generate shape function plots & HTML

Lihat [playground/ebm/README.md](playground/ebm/README.md) untuk detail.

### Preprocessor (LabelEncoder + fillna pipeline)

```bash
cd playground/preprocessing
../.venv/bin/python fit_preprocessor.py                        # identity-only mode
../.venv/bin/python fit_preprocessor.py \
  --full /path/to/train_transaction.csv                        # full features
```

Output: `playground/preprocessing/preprocessor.pkl` (~45 KB).

Lihat [playground/preprocessing/README.md](playground/preprocessing/README.md).

### LGBM predictor

Saat ini training-nya di notebook (belum di-extract jadi `.py` standalone):
[playground/2025-05-01_Model.ipynb](playground/2025-05-01_Model.ipynb).

Butuh `train_transaction.csv` (~470 MB dari Kaggle) — tidak ada di repo.
Output: `playground/lgbm_tuning.pkl` (~35 MB) yang dicopy ke
`ml-service/models/lgbm.pkl`.

```bash
# Setelah menjalankan notebook
cp playground/lgbm_tuning.pkl ml-service/models/lgbm.pkl
```

### Deploy model baru ke ml-service

Setelah retraining, update artifacts dan restart service:

```bash
# Copy artifact baru
cp playground/ebm/ebm_model.pkl ml-service/models/ebm.pkl
cp playground/lgbm_tuning.pkl ml-service/models/lgbm.pkl
cp playground/preprocessing/preprocessor.pkl ml-service/models/preprocessor.pkl

# (Opsional) bump version di config.yaml supaya audit trail jelas
# Edit ml-service/config.yaml: predictor.version: "1.1"

# Rebuild + restart
docker compose up -d --build
```

### Feature store re-materialize (setelah update training data)

```bash
cd feature-store
../playground/.venv/bin/python seed_data.py     # CSV → parquet
./apply.sh                                       # registry
./materialize.sh                                 # online store
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
