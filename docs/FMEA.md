# FMEA — Failure Mode and Effects Analysis

**Sistem:** Real-Time Fraud Detection — IF5251 Group Project
**Versi dokumen:** 1.0 — 2026-05-10

Memenuhi spec IF5251 nomor **Requirement 1c**.

## Metodologi

Setiap potensi kegagalan komponen dievaluasi dengan tiga skor (skala 1–10):

- **S (Severity):** dampak ke user/sistem jika terjadi
- **P (Probability):** seberapa mungkin terjadi dalam operasi normal
- **D (Detection difficulty):** seberapa sulit dideteksi sebelum berdampak

**RPN = S × P × D**. Prioritas mitigasi:
- RPN ≥ 200: **kritis**, mitigasi wajib sebelum produksi
- RPN 100–199: **tinggi**, mitigasi terjadwal
- RPN < 100: **rendah**, monitor saja

## Komponen yang dianalisis

```
Client → API Gateway → ml-service ──┬─► Feast online store
                          │         │
                          ├─► LGBM model (predictor)
                          ├─► EBM model (explainer)
                          └─► Great Expectations validator

Reviewer UI (Streamlit) → ml-service
Cron → Evidently → drift report
Cron → FairLearn → fairness report
CI/CD GitHub Actions → tests + docker build
```

---

## 1. ml-service (FastAPI)

| ID | Failure mode | Effect | S | P | D | RPN | Mitigation |
|----|--------------|--------|---|---|---|-----|-----------|
| ML-1 | Model file `.pkl` corrupt / missing | Service crash di startup | 8 | 2 | 2 | 32 | Health check + Dockerfile COPY validates files exist; regenerate dari training script |
| ML-2 | Cold start > 5 detik (load LGBM 35MB) | First request timeout | 4 | 6 | 3 | 72 | Pre-warm via `/health` di Kubernetes readinessProbe; gunakan startup probe dengan timeout 30s |
| ML-3 | OOM saat load LGBM + EBM bersamaan | Pod restart loop | 8 | 3 | 2 | 48 | Set memory limit 1Gi, monitor RSS; consider model quantization untuk LGBM |
| ML-4 | Race condition saat config reload | Server return inconsistent prediction | 6 | 2 | 8 | 96 | Tidak ada hot-reload; restart untuk swap model |
| ML-5 | Adapter raise exception untuk input edge case | 500 ke client | 5 | 5 | 4 | 100 | GX validation di hulu (3a); structured error logging; fallback ke default proba |
| ML-6 | sklearn version mismatch (training 1.3 vs serving 1.8) | LabelEncoder warning, inference tetap jalan tapi nilai bisa beda | 7 | 7 | 9 | **441** | **KRITIS** — pin sklearn version di requirements.txt; persist LabelEncoder pkl bersama model |

## 2. LGBM model (predictor)

| ID | Failure mode | Effect | S | P | D | RPN | Mitigation |
|----|--------------|--------|---|---|---|-----|-----------|
| LGBM-1 | Concept drift — pola fraud baru tidak terdeteksi | False negative rate naik | 9 | 8 | 7 | **504** | **KRITIS** — Evidently drift report mingguan; trigger continuous training (Ops 4a) |
| LGBM-2 | Adversarial input dengan small perturbation flip prediksi | Fraud lolos saringan | 8 | 4 | 6 | 192 | Adversarial test (QA 3b); ensemble prediction; threshold ramp-up untuk borderline |
| LGBM-3 | Bias terhadap group sensitif (DeviceType, Browser) | Disparate impact, kompliance issue | 7 | 9 | 5 | 315 | **KRITIS** — FairLearn audit (RAI 5a); reweight training; threshold per group |
| LGBM-4 | Model output probabilistik miscalibrated | Threshold 0.5 tidak optimal lagi | 6 | 5 | 6 | 180 | Calibration plot reguler; tune threshold via Youden index dari validation |

## 3. EBM model (explainer)

| ID | Failure mode | Effect | S | P | D | RPN | Mitigation |
|----|--------------|--------|---|---|---|-----|-----------|
| EBM-1 | Distillation drift — student menyimpang dari teacher | Explanation tidak konsisten dengan prediksi LGBM | 6 | 6 | 8 | 288 | **KRITIS** — fidelity Pearson check setiap retrain (target ≥ 0.95); distill ulang setelah LGBM diupdate |
| EBM-2 | Shape function tidak monotonic untuk fitur yang seharusnya monotonic | Explanation membingungkan reviewer | 4 | 4 | 9 | 144 | Visual inspection plot; tambah constraint monotonic di EBM hyperparameter saat training |
| EBM-3 | Hanya 25 fitur (subset) — tidak menjelaskan signal V/C/D | Reviewer tidak punya konteks penuh | 5 | 10 | 1 | 50 | Dokumen-kan limitasi; saat data lengkap tersedia, retrain EBM dengan 424 fitur |

## 4. Feast feature store

| ID | Failure mode | Effect | S | P | D | RPN | Mitigation |
|----|--------------|--------|---|---|---|-----|-----------|
| FS-1 | Online store SQLite corrupt | `/predict_by_id` 500/404 untuk semua entity | 9 | 2 | 3 | 54 | Re-materialize dari parquet; backup parquet ke external storage |
| FS-2 | TransactionID belum di-materialize | 404 false negative | 6 | 7 | 2 | 84 | Materialize-incremental setelah ingestion; pre-flight check di health endpoint |
| FS-3 | Stale data — fitur tidak refresh setelah training data update | Serving pakai feature value lama | 7 | 6 | 6 | 252 | **KRITIS** — re-materialize on schedule; TTL appropriate per FeatureView |
| FS-4 | Schema drift di parquet (kolom hilang/rename) | Apply gagal atau hasil lookup salah | 8 | 3 | 5 | 120 | GX schema check sebelum materialize; CI test dengan schema reference |

## 5. Great Expectations validator

| ID | Failure mode | Effect | S | P | D | RPN | Mitigation |
|----|--------------|--------|---|---|---|-----|-----------|
| GX-1 | False negative — invalid input lolos validasi | Model dipanggil dengan garbage, prediksi tidak reliable | 6 | 4 | 7 | 168 | Tambah expectation incrementally berdasarkan field yang ditemukan invalid; review reject log mingguan |
| GX-2 | False positive — valid input ditolak (412 ke user) | UX terganggu, drop rate naik | 4 | 5 | 3 | 60 | Loosen `mostly` threshold; expectation per kolom punya escape hatch (`mostly=0.99`) |
| GX-3 | Suite tidak update saat schema berubah | Validation pakai schema lama | 5 | 4 | 6 | 120 | CI/CD run `validate_dataset.py` setiap PR; require manual approval untuk add expectation |
| GX-4 | Performance — validate per request menambah latency | p99 latency meningkat | 4 | 7 | 2 | 56 | Cache singleton validator; benchmark < 5ms/request; opsi disable di high-throughput path |

## 6. Streamlit UI

| ID | Failure mode | Effect | S | P | D | RPN | Mitigation |
|----|--------------|--------|---|---|---|-----|-----------|
| UI-1 | Reviewer accidentally klik wrong button | Audit log salah | 5 | 4 | 9 | 180 | Add confirmation dialog untuk Confirm fraud; allow undo via "edit decision" |
| UI-2 | ML_SERVICE_URL salah (network split) | UI tidak bisa score | 7 | 2 | 1 | 14 | Sidebar tampilkan service health; show error jelas |
| UI-3 | decisions.jsonl growing unbounded | Disk full | 4 | 8 | 5 | 160 | Log rotation; archive ke object storage harian |

## 7. Server / infrastruktur

| ID | Failure mode | Effect | S | P | D | RPN | Mitigation |
|----|--------------|--------|---|---|---|-----|-----------|
| INF-1 | Server ITB ephemeral — crash total | Semua artifact (online store, decisions log) hilang | 9 | 6 | 1 | 54 | Backup script ke external (GDrive/S3) tiap 6 jam; immutable infra — semua artifact bisa di-regenerate dari source |
| INF-2 | GPU P100 OOM saat batch inference besar | Predict latency naik drastis | 6 | 3 | 4 | 72 | Batch size limit di adapter; LGBM CPU-bound — GPU hanya untuk training |
| INF-3 | Network partition antara ml-service ↔ feature-store | `/predict_by_id` 503 | 7 | 2 | 2 | 28 | Currently same-host; saat di K3s, gunakan service discovery + retry budget |
| INF-4 | Adversarial DDoS pada `/predict` | Service unresponsive | 8 | 4 | 4 | 128 | Rate limiting per IP di API Gateway; circuit breaker di adapter |

## 8. Privacy & Security

| ID | Failure mode | Effect | S | P | D | RPN | Mitigation |
|----|--------------|--------|---|---|---|-----|-----------|
| SEC-1 | Model inversion attack — leakage training data via prediction patterns | Privacy breach | 9 | 3 | 9 | 243 | **KRITIS** — rate limit + query budget per user; differential privacy noise di output (RAI 5c) |
| SEC-2 | Predictions log mengandung PII | Compliance issue | 8 | 5 | 5 | 200 | **KRITIS** — hash transaction_id sebelum log; encrypt log di rest |
| SEC-3 | Endpoint `/predict` terbuka tanpa auth | Abuse | 7 | 8 | 1 | 56 | Tambah API key di middleware; rate limit |

---

## Ringkasan prioritas mitigasi (RPN ≥ 200)

| RPN | ID | Failure mode | Mitigation status |
|-----|----|--------------|----|
| **504** | LGBM-1 | Concept drift | Partial — Evidently jalan; continuous training belum |
| **441** | ML-6 | sklearn version mismatch | Open — pin version + persist encoder |
| **315** | LGBM-3 | Bias terhadap group sensitif | Detected (FairLearn audit menunjukkan disparate impact); mitigation belum |
| **288** | EBM-1 | Distillation drift | Partial — fidelity check manual; otomatisasi belum |
| **252** | FS-3 | Feast stale data | Partial — `materialize.sh` ada, schedule belum |
| **243** | SEC-1 | Model inversion | Open |
| **200** | SEC-2 | PII di log | Open |

## Decision matrix — apa yang harus dikerjakan dulu

1. **ML-6 (sklearn pin)**: trivial, langsung kerjakan
2. **LGBM-1 + EBM-1 (drift + distillation)**: butuh CI/CD scheduled job
3. **LGBM-3 (bias)**: butuh threshold per group atau reweighting — research time
4. **SEC-1, SEC-2 (privacy)**: butuh rate limiting + log scrubbing — implementasi medium

## Glossary

- **RPN** — Risk Priority Number, perkalian S × P × D
- **Continuous training** — auto-retrain saat data drift terdeteksi
- **Disparate impact** — selection rate berbeda signifikan antar group sensitif
- **Distillation drift** — student model menyimpang dari teacher seiring waktu
- **Model inversion** — attack untuk recover training data dari model output
