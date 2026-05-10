# Preprocessing Module

Reusable preprocessing pipeline. Encapsulates steps from
`2025-03-15_Preprocessing.ipynb` jadi modul yang bisa di-fit sekali,
disimpan ke disk, dan dipakai ulang.

## Files

| File | Isi | Sharable? |
|------|-----|-----------|
| `preprocess.py` | Class `Preprocessor` (kode) | Ya — copy ke teman |
| `fit_preprocessor.py` | Script untuk fit & save | Ya |
| `preprocessor.pkl` | Fitted state (encoders, kolom, threshold) | **Ya — yang ini di-share ke teman** |
| `requirements.txt` | Dependencies | Ya |

## What's inside `preprocessor.pkl`

```python
Preprocessor(
    drop_columns=['id_07','id_21','id_22','id_23','id_24',
                  'id_25','id_26','id_27','TransactionID'],
    txn_amt_cap=20000,
    null_token='null',
    numeric_fillna=-999,
    encoders={
        'id_12': LabelEncoder(...),       # 15 LabelEncoder
        'id_15': LabelEncoder(...),
        'id_30': LabelEncoder(...),       # OS values mapped to int
        'id_31': LabelEncoder(...),       # browser values mapped to int
        'DeviceInfo': LabelEncoder(...),
        ...
    },
    categorical_cols=['id_12','id_15','id_16','id_28','id_29',
                      'id_30','id_31','id_33','id_34','id_35',
                      'id_36','id_37','id_38','DeviceType','DeviceInfo'],
    numeric_cols=['id_01','id_02','id_03','id_04','id_05','id_06',
                  'id_08','id_09','id_10','id_11','id_13','id_14',
                  'id_17','id_18','id_19','id_20','id_32'],
    feature_names=[...32 column names...],
    fitted=True,
)
```

Ukuran ~45 KB. Yang di-pickle: state hasil fit, bukan data training.

## Usage

```python
from preprocess import Preprocessor
import pandas as pd

# === Sekali, di awal proyek ===
df = pd.read_csv("train_identity.csv")
pre = Preprocessor()
df_processed = pre.fit_transform(df)
pre.save("preprocessor.pkl")

# === Kapan saja, di mana saja (notebook lain, script lain, mesin teman) ===
pre = Preprocessor.load("preprocessor.pkl")
new_data = pd.read_csv("data_baru.csv")
new_processed = pre.transform(new_data)
# new_processed siap untuk model.predict()
```

## Apa yang dilakukan pipeline

1. **Rename kolom**: `id-XX` → `id_XX` (test set Kaggle pakai dash)
2. **Drop kolom**: `id_07`, `id_21`–`id_27`, `TransactionID`
3. **Drop outlier**: `TransactionAmt > 20000` (saat fit saja, default tidak saat transform)
4. **Fillna**: kategorikal → `'null'`, numerik → `-999`
5. **Label encoding** untuk kategorikal — kategori unseen di-treat sebagai `'null'`

## Cara share ke teman sekelompok

### Opsi 1: Git (paling reliable)
```bash
git add preprocessing/
git commit -m "Add reusable preprocessing module"
git push
# Teman: git pull
```

### Opsi 2: Zip + cloud drive
```bash
cd playground
zip -r preprocessing.zip preprocessing/
# Upload ke Drive/Dropbox, share link
```

### Opsi 3: Cuma file `.pkl` (kalau teman udah punya `preprocess.py`)
File `preprocessor.pkl` sendirian (~45 KB) bisa dikirim via WhatsApp / Discord.
Tapi teman harus punya `preprocess.py` di path yang sama saat `joblib.load()`,
karena pickle butuh class `Preprocessor` untuk reconstruct objek.

## Catatan penting

- Versi sklearn harus kompatibel antara mesin yang fit dan yang load.
  Kalau teman pakai sklearn versi beda, mungkin muncul warning
  `InconsistentVersionWarning`.
- File `preprocessor.pkl` saat ini di-fit pada `train_identity.csv` saja
  (32 kolom). Kalau punya `train_transaction.csv`, jalankan ulang:
  ```bash
  python fit_preprocessor.py --full ../train_transaction.csv
  ```
  Akan menghasilkan preprocessor untuk full 425 kolom.
