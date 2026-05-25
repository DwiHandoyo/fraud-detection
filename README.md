# Fraud Detection — IF5251 Group Project

Real-Time Fraud Detection system untuk mata kuliah **Engineering AI-Enabled
Systems (IF5251)** di ITB. Dataset: IEEE-CIS Fraud Detection (Vesta + Kaggle).

## Komponen

| # | Folder | Tujuan | Spec |
|---|--------|--------|------|
| 1 | [ml-service/](ml-service/) | FastAPI inference + explanation, adapter pattern model-agnostic | Arch 2b, RAI 5b |
| 2 | [feature-store/](feature-store/) | Feast feature store (online SQLite + offline parquet) | Arch 2a |
| 3 | [postgres/](postgres/) | PostgreSQL — audit storage untuk predictions + decisions | Req 1a |
| 4 | [data-validation/](data-validation/) | Great Expectations suite (dataset + request) | QA 3a |
| 5 | [ui/](ui/) | Streamlit multi-page console (4 pages) | Arch 2c |
| 6 | [monitoring/](monitoring/) | Evidently drift dashboard | Ops 4b |
| 7 | [fairness/](fairness/) | FairLearn audit (DeviceType, OS, Browser) | RAI 5a |
| 8 | [qa-tests/](qa-tests/) | Adversarial + locust load test | QA 3b, 3c |
| 9 | [docs/FMEA.md](docs/FMEA.md) | Failure Mode and Effects Analysis | Req 1c |
| 10 | [.github/workflows/](.github/workflows/) | CI/CD (lint + test + build) | Ops 4a |
| 11 | [playground/](playground/) | EDA notebooks, EBM training, original LGBM | (research) |

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
| `train_identity.csv` | `playground/train_identity.csv` | ~25 MB | committed | [Kaggle IEEE-CIS](https://www.kaggle.com/c/ieee-fraud-detection/data) |
| `identity.parquet` (Feast offline) | `feature-store/feature_repo/data/identity.parquet` | ~5.7 MB | committed | Regen dari `train_identity.csv` via [seed_data.py](feature-store/seed_data.py) |
| `online_store.db` (Feast online) | `feature-store/feature_repo/data/online_store.db` | ~965 MB | gitignored (>GitHub limit) | Regen via `./materialize.sh` |
| `registry.db` (Feast metadata) | `feature-store/feature_repo/data/registry.db` | 3 KB | gitignored | Regen via `./apply.sh` |
| `train_transaction.csv` | (tidak ada di repo) | ~470 MB | download manual | [Kaggle IEEE-CIS](https://www.kaggle.com/c/ieee-fraud-detection/data), wajib untuk retrain LGBM |
| `sample_10rows.csv` | `playground/sample_10rows.csv` | 2 KB | committed | Sample untuk smoke test |
| Identity reports (HTML) | `data-validation/reports/`, `monitoring/reports/`, `fairness/reports/` | varies | gitignored | Regen via masing-masing `*.py` script |

### Try-it-out walkthrough (5 menit)

Untuk reviewer:

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

### Feast di training vs production

Definisi `FeatureView` di
[feature-store/feature_repo/features.py](feature-store/feature_repo/features.py)
dipakai dua jalur:

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

Definisi fitur sama, jadi tidak ada training/serving skew (spec Arch 2a).

### Training scripts

Ada **3 artifact** yang dilatih, di-ranking dari yang paling sering diretrain:

| Artifact | Script training | Butuh data lengkap? | Output |
|----------|----------------|---------------------|--------|
| EBM (explainer) | [`playground/ebm/train_ebm.py`](playground/ebm/train_ebm.py) | tidak (mode distillation) | `ebm_model.pkl` |
| LGBM (predictor) | [`playground/lgbm/train_lgbm.py`](playground/lgbm/train_lgbm.py) | ya, butuh `train_transaction.csv` | `lgbm_model.pkl` + `label_encoders.pkl` |
| Preprocessor | [`playground/preprocessing/fit_preprocessor.py`](playground/preprocessing/fit_preprocessor.py) | identity-only atau full | `preprocessor.pkl` |

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

Status saat ini:
- Feast offline store sudah ready, parquet ada di
  `feature-store/feature_repo/data/identity.parquet`
- `playground/2025-05-01_Model.ipynb` (LGBM training) saat ini baca langsung
  dari CSV, belum migrasi ke `get_historical_features()`
- `playground/ebm/train_ebm.py` (EBM training) juga baca langsung dari CSV
- Refactor opportunity: ubah kedua training jalur supaya pakai
  `get_historical_features()` untuk demonstrasi konsistensi training/serving
  yang lebih meyakinkan di laporan.

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

Standalone script: [playground/lgbm/train_lgbm.py](playground/lgbm/train_lgbm.py).
Butuh `train_transaction.csv` (~470 MB dari
[Kaggle IEEE-CIS](https://www.kaggle.com/c/ieee-fraud-detection/data)) — tidak ada di repo.

```bash
cd playground/lgbm
../.venv/bin/python -m pip install -r requirements.txt

# Smoke test (~30 detik, n_estimators=200)
../.venv/bin/python train_lgbm.py \
  --transaction-csv /path/to/train_transaction.csv \
  --quick

# Full training (~10–15 menit, val AUC ~0.92)
../.venv/bin/python train_lgbm.py \
  --transaction-csv /path/to/train_transaction.csv
```

Output: `lgbm_model.pkl` + `label_encoders.pkl` (untuk consistency
inference/training) + `metrics.json` + `feature_importance.csv`.
Detail di [playground/lgbm/README.md](playground/lgbm/README.md).

Notebook asli [playground/2025-05-01_Model.ipynb](playground/2025-05-01_Model.ipynb)
tetap ada untuk EDA / experimentation.

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

### Continuous training (human-in-the-loop feedback)

Pipeline lengkap dari koreksi reviewer → retrain → deploy. Mix dari step
otomatis (DB-driven) dan manual (deliberate, supaya reviewer bisa kontrol).

#### Alur lengkap (post Bulk Audit)

```
[Page 5 Bulk Predict] upload CSV
      │ AUTO
      ▼
bulk_predictions table — status=completed, predicted_label, fraud_proba
      │
      ▼
[Page 7 Bulk Audit] reviewer centang rows + klik "Mark as FRAUD/LEGIT/BLOCKED"
      │ AUTO (instant DB write)
      ▼
bulk_predictions.user_label, user_note, labeled_at
      │
      ├──► AUTO ──► [Page 6 Bulk Dashboard] pie chart "User label breakdown"
      │
      │ MANUAL (user buka Page 8)
      ▼
[Page 8 Continuous Training]
   • inventory: berapa labeled rows
   • slider parameters (recent_days, history_sample, min_new_labels)
   • klik "Show retrain command" → UI generate command + copy hint
      │
      │ MANUAL (copy-paste ke host terminal — UI image tidak punya lightgbm)
      ▼
$ export DATABASE_URL=postgresql+psycopg2://fraud:fraud_demo_only@localhost:5432/fraud
$ cd playground/lgbm
$ ../.venv/bin/python retrain_from_feedback.py --recent-days 30 ...
      │ AUTO (dalam script)
      ▼
1. SELECT labeled rows: NEW (last 30 days, 100%) + OLD (>30 days, 20% sample)
2. Expand request_payload JSONB → feature columns, user_label → isFraud
3. Reuse train_lgbm.preprocess + train (LabelEncoder fit, LGBM fit, val AUC)
4. Save artifacts:
     ml-service/models/lgbm_v{timestamp}.pkl
     ml-service/models/label_encoders_v{timestamp}.pkl
     INSERT INTO training_jobs (status='trained', val_auc, n_new, n_old, ...)
      │
      ▼
Script print: "To promote: edit ml-service/config.yaml..."
      │
      │ MANUAL (kembali ke Page 8)
      ▼
[Page 8 — section "Promote a trained model"]
   • Dropdown: training_jobs WHERE status='trained'
   • Klik "Promote to Champion"
      │ AUTO (DB only)
      ▼
UPDATE training_jobs SET status='promoted', promoted_at=NOW()
      │
      │ MANUAL (UI tampilkan instruksi)
      ▼
$ # Edit ml-service/config.yaml:
$ #   predictor.path: models/lgbm_v{timestamp}.pkl
$ #   predictor.version: "v{timestamp}"
$ docker compose restart ml-service
      │ AUTO (ml-service startup)
      ▼
ml-service load model baru → /predict serve versi baru
```

#### Ringkasan otomatis vs manual

| Step | Otomatis | Manual |
|------|----------|--------|
| Bulk Audit klik button → DB write | ya | |
| Bulk Dashboard refresh visualization | ya | |
| Trigger retrain | | ya, user copy command |
| Train LGBM + save versioned pkl | ya (dalam script) | |
| INSERT training_jobs row | ya | |
| Promote ke champion (DB update) | ya (button) | |
| Edit config.yaml + restart ml-service | | ya, user manual |
| ml-service load model baru | ya (on startup) | |

#### Yang sengaja TIDAK otomatis

- Auto-trigger retrain saat N labels terkumpul. Reviewer kontrol kapan.
- Auto-promote saat AUC lebih baik. Perlu sanity check manusia.
- Hot-reload model tanpa restart. Keep deployment audit-trail jelas.
- Shadow mode (paralel run challenger). Future work, disebut di FMEA.

#### Windowing strategy

Sliding window dengan history sample:

```
NEW (last 30 days):       100% used   → adapt ke concept drift
OLD (>30 days):           20% sample  → hindari catastrophic forgetting
```

Knob `--recent-days` dan `--history-sample-rate` di UI Page 8 atau CLI.

#### Sumber fitur saat retrain

Dari `bulk_predictions.request_payload` (JSONB snapshot saat user upload CSV),
**bukan** Feast online store. Trade-off ini disengaja untuk simplicity demo —
production sungguhan pakai Feast `get_historical_features()` dengan
event_timestamp untuk point-in-time correctness.

#### Quick reference commands

```bash
# Retrain (host terminal, butuh playground/.venv dengan lightgbm)
export DATABASE_URL=postgresql+psycopg2://fraud:fraud_demo_only@localhost:5432/fraud
cd playground/lgbm
../.venv/bin/python retrain_from_feedback.py \
  --recent-days 30 \
  --history-sample-rate 0.20 \
  --min-new-labels 20 \
  --quick

# Promote (di Page 8 UI) lalu activate:
# Edit ml-service/config.yaml predictor.path → models/lgbm_v{ts}.pkl
docker compose restart ml-service
```

Detail lengkap: [playground/lgbm/retrain_from_feedback.py](playground/lgbm/retrain_from_feedback.py)
dan [ui/pages/8_Continuous_Training.py](ui/pages/8_Continuous_Training.py).

## Arsitektur

```
                 ┌─────────────────────────┐
                 │  Streamlit UI (ui/)     │  http://localhost:8501
                 │  3 pages: What-if,      │
                 │  Audit, Status          │
                 └────────┬────────────────┘
                          │ HTTP
                          ▼
┌──────────────┐   ┌──────────────────┐   ┌──────────────────┐
│ Great        │◄──│  ml-service      │──►│ Feast online     │
│ Expectations │   │  (FastAPI)       │   │ store (SQLite)   │
│ validator    │   │                  │   │                  │
│              │   │  ┌────────────┐  │   └────────┬─────────┘
│ data-        │   │  │ LGBM       │  │            │
│ validation/  │   │  │ adapter    │  │            ▼
└──────────────┘   │  │ (predict)  │  │   ┌──────────────────┐
                   │  └────────────┘  │   │ Feast offline    │
                   │  ┌────────────┐  │   │ (parquet)        │
                   │  │ EBM        │  │   │ feature-store/   │
                   │  │ adapter    │  │   └──────────────────┘
                   │  │ (explain)  │  │
                   │  └────────────┘  │
                   └────────┬─────────┘
                            │ insert
                            │
            ┌───────────────┴───────────────┐
            │                               │
            ▼                               ▼
   ┌──────────────────┐         ┌──────────────────────────┐
   │ Predictions log  │         │  PostgreSQL (postgres/)  │
   │ JSONL (volume)   │         │  - predictions table     │
   └────────┬─────────┘         │  - decisions table       │
            │                   └──────────┬───────────────┘
            ▼                              │
   ┌──────────────────┐                    │
   │ Evidently        │         ┌──────────┴──────┐
   │ (monitoring/)    │         │ Audit Log page  │
   └──────────────────┘         │ (UI)            │
                                └─────────────────┘
                                         ▲
                                         │
                                Streamlit UI insert
                                decision via api_client
```

Graceful fallback: kalau PostgreSQL down, ml-service + UI tetap berjalan
(prediksi via JSONL, audit log via JSONL).

## Spec coverage

| Spec | Status | Komponen |
|------|--------|----------|
| Req 1a, Komponen non-AI (DB, UI, API) | done | [postgres/](postgres/) + [ui/](ui/) + [ml-service/](ml-service/) |
| Req 1c, FMEA | done | [docs/FMEA.md](docs/FMEA.md) |
| Arch 2a, Feature Store (Feast) | done | [feature-store/](feature-store/) |
| Arch 2b, Microservice + container | done | [ml-service/](ml-service/) |
| Arch 2c, Human review UI | done | [ui/](ui/) |
| QA 3a, Data validation | done | [data-validation/](data-validation/) |
| QA 3b, Adversarial test | done | [qa-tests/adversarial.py](qa-tests/adversarial.py) |
| QA 3c, Load test | done | [qa-tests/load_test.py](qa-tests/load_test.py) |
| Ops 4a, CI/CD + continuous training | done | [.github/workflows/ci.yml](.github/workflows/ci.yml) + [playground/lgbm/retrain_from_feedback.py](playground/lgbm/retrain_from_feedback.py) |
| Ops 4b, Monitoring (Evidently) | done | [monitoring/](monitoring/) |
| RAI 5a, Fairness audit | done | [fairness/](fairness/) |
| RAI 5b, Explainability | done | [ml-service/app/adapters.py](ml-service/app/adapters.py) `EBMAdapter.explain` |
| RAI 5c, Encryption + model inversion | partial | dijelaskan di FMEA, mitigasi belum diimplementasi |

## Model architecture

**Single model: `lgbm_best.pkl`** — LightGBM dengan 107 fitur (hasil retuning
dari notebook), berfungsi sebagai **predictor sekaligus explainer**.

- `/predict` dan `/predict_by_id` → `LGBMAdapter.predict_proba(X)`
- `/explain` dan `/explain_by_id` → `LGBMAdapter.explain(X)` via SHAP TreeExplainer

Karena predictor dan explainer pakai model yang sama, **fraud_proba dari
`/predict` dan `/explain` identik** untuk input yang sama (no LGBM↔EBM drift).

**Alternatif tersedia:**
- `EBMAdapter` di [adapters.py](ml-service/app/adapters.py) — glass-box,
  artifact di [playground/ebm/](playground/ebm/). Swap dengan ubah `config.yaml`
  → `type: ebm`.
- `lgbm.pkl` lama (424 fitur) — tersedia di `ml-service/models/` sebagai backup.

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
