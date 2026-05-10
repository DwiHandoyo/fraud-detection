# fairness — FairLearn audit

Memenuhi spec IF5251 nomor **Responsible AI 5a**: audit kelompok sensitif.

## Apa yang diaudit

LGBM fraud model di-evaluasi untuk *disparate impact* terhadap 3 fitur sensitif
yang dianggap proxy demografis di dataset IEEE-CIS:

| Fitur sensitif | Proxy untuk | Kategori |
|----------------|-------------|----------|
| `DeviceType` | Kelas ekonomi (mobile-only user vs desktop) | mobile / desktop / unknown |
| `OS_family` | Device price tier (iOS premium vs Windows mid) | Apple / Android / Windows / Linux / other |
| `Browser_family` | Tech-savviness / corporate vs personal | chrome / safari / firefox / edge / ie / opera / samsung / other |

## Metrik

- **Demographic Parity Difference (DPD)** = |max - min| selection rate antar group.
  Acceptable: ≤ 0.10
- **Demographic Parity Ratio (DPR)** = min / max selection rate.
  Acceptable (four-fifths rule): ≥ 0.80

## Caveat

`train_transaction.csv` tidak tersedia di repo, jadi label `isFraud` asli
tidak ada. Audit menggunakan **prediksi LGBM** sebagai proxy label
(threshold 0.5). Ini valid untuk mendeteksi **bias dalam model**, tapi tidak
bisa mendeteksi **bias dalam data** (yang butuh ground truth).

## Run

```bash
cd fairness
../playground/.venv/bin/python -m pip install -r requirements.txt
../playground/.venv/bin/python audit.py
# Output: reports/fairness_metrics.json + reports/fairness_summary.html
```

## Hasil temuan (last run, 50k sample)

| Sensitif | DPD | DPR | Verdict |
|----------|-----|-----|---------|
| DeviceType | 0.539 | 0.014 | **FAIL** — disparitas substansial |
| OS_family | 0.898 | 0.102 | **FAIL** — model sangat bias |
| Browser_family | 0.730 | 0.033 | **FAIL** — disparitas substansial |

Contoh disparitas konkret:
- **Browser_family**: Safari user di-flag fraud 75.5% — Edge user 2.8%. Rasio 27×.
- **OS_family**: iOS / Android / Linux user 100% di-flag — Windows 44.2% — unknown OS 10.2%.

## Implikasi untuk laporan

1. **Model tidak fair** terhadap dimensi browser/OS. Ini perlu dimitigasi:
   - Reweight training samples by sensitive group
   - Threshold per group (post-processing)
   - FairLearn `ExponentiatedGradient` atau `GridSearch` dengan constraint
2. Pada saat dataset asli (dengan isFraud) tersedia, ulangi audit pakai
   ground truth untuk membedakan bias model vs bias data.

## Limitasi

- "Selection rate per group" ekstrim (mis. iOS 100%) didorong oleh fitur lain
  yang berkorelasi (id_31 = mobile safari → id_30 = iOS). LGBM mungkin belajar
  shortcut "iOS → fraud" karena kombinasi feature lain di training set.
- Audit hanya sebagian (LGBM only). EBM tidak diaudit — mendukung audit serupa
  dengan trivial ekstensi.
