"""
Automated Data Ingestion and Feature Engineering Pipeline for Feast.

This script checks for an existing preprocessed Parquet file. If it exists,
it skips the heavy preprocessing step. Otherwise, it reads raw CSVs, applies 
the feature engineering pipeline, and saves the output to identity.parquet.

Output: feature_repo/data/identity.parquet
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import os
import gdown
from sklearn.preprocessing import LabelEncoder

HERE = Path(__file__).resolve().parent
PLAYGROUND = HERE.parent / "playground"

# Source RAW CSV path definition
TRANSACTION_CSV = PLAYGROUND / "train_transaction.csv"
IDENTITY_CSV = PLAYGROUND / "train_identity.csv"

# Destination path for Feast Offline Store (Updated filename)
OUT_PARQUET = HERE / "feature_repo" / "data" / "identity.parquet"

# Baseline time anchor agreed for the transaction relative time simulation
REFERENCE_DATE = datetime(2025, 1, 1)


def reduce_groups(grps: list[list[str]], df: pd.DataFrame) -> list[str]:
    """Identify columns with the highest number of unique values in each group."""
    use = []
    for col in grps:
        max_unique = 0
        max_index = 0
        for i, c in enumerate(col):
            n = df[c].nunique()
            if n > max_unique:
                max_unique = n
                max_index = i
        use.append(col[max_index])
    return use


def main() -> None:
    # check if the preprocessed Parquet file already exists
    if OUT_PARQUET.exists():
        print(f"[seed] Great! '{OUT_PARQUET.name}' already exists.")
        print(f"[seed] Skipping raw CSV processing. Feast will read directly from the existing file.")
        return

    print(f"[seed] '{OUT_PARQUET.name}' NOT found. Starting pipeline from raw CSVs...")
    print(f"[seed] Reading raw files from {PLAYGROUND}...")
    
    # Define Google Drive file ID and construct the download URL
    url_train_transaction = '1-LA_hivwi-LTmk3Il6zWc2SlWkw63lIp'
    gdrive_url = f'https://drive.google.com/uc?id={url_train_transaction}'
    
    # Check if already exists
    if not os.path.exists(TRANSACTION_CSV):
        print(f"[info] {TRANSACTION_CSV} Not found. Starting download from Google Drive...")
        try:
            # Download langsung diarahkan ke path TRANSACTION_CSV tujuan
            # gdown.download(gdrive_url, TRANSACTION_CSV, quiet=False)
            gdown.download(gdrive_url, str(TRANSACTION_CSV), quiet=False)
            print("[success] Download train_transaction.csv selesai.")
        except Exception as download_error:
            print(f"[error] Gagal mendownload file dari Google Drive. Details: {download_error}")
            return

    # Load to dataframes
    try:
        train_transaction = pd.read_csv(TRANSACTION_CSV)
        train_identity = pd.read_csv(IDENTITY_CSV)
        print("[success] Successfully loaded train_transaction and train_identity.")
        
        # Lanjutkan proses setelah ini
        
    except FileNotFoundError as e:
        print(f"[error] Failed to locate raw CSV files in {PLAYGROUND}. Details: {e}")
        return
        
    print(f"[seed] Loaded transaction: {train_transaction.shape}, identity: {train_identity.shape}")

    # --- 1. MERGE & FEAST TIMESTAMP INJECTION ---
    df_train = pd.merge(train_transaction, train_identity, on='TransactionID', how='left')
    df_train['event_timestamp'] = REFERENCE_DATE + pd.to_timedelta(df_train['TransactionDT'], unit='s')

    # --- 2. DROPPING DUPLICATES & OUTLIERS ---
    duplicate_cols = df_train.columns.drop('TransactionID')
    df_train = df_train.drop_duplicates(subset=duplicate_cols, keep='first').reset_index(drop=True)
    
    drop_column = ['id_07', 'id_08', 'id_21', 'id_22', 'id_23', 'id_24', 'id_25', 'id_26', 'id_27']
    df_train.drop(columns=drop_column, errors='ignore', inplace=True)
    df_train = df_train.drop(df_train[df_train['TransactionAmt'] > 30000].index).reset_index(drop=True)

    # --- 3. V-FEATURES REDUNDANCY REDUCTION ---
    v_missing_cols = [col for col in df_train.columns if col.startswith('V') and df_train[col].isnull().any()]
    v_nan_dict = {}
    for col in v_missing_cols:
        count = df_train[col].isnull().sum()
        v_nan_dict.setdefault(count, []).append(col)

    v_groups_to_reduce = [v for k, v in v_nan_dict.items() if len(v) > 1]
    all_v_in_groups = [col for group in v_groups_to_reduce for col in group]

    v_cols_to_keep = reduce_groups(v_groups_to_reduce, df_train)
    v_cols_to_drop = list(set(all_v_in_groups) - set(v_cols_to_keep))
    df_train.drop(columns=v_cols_to_drop, inplace=True)

    # --- 4. TIME & D-FEATURES NORMALIZATION ---
    df_train['Transaction_Day'] = df_train['TransactionDT'] / 86400

    for i in range(1, 16):
        col = f'D{i}'
        if col in df_train.columns:
            df_train[f'{col}_norm'] = df_train[col] - df_train['Transaction_Day']
            df_train.drop(columns=[col], inplace=True)
            df_train[f'{col}_norm'] = df_train[f'{col}_norm'].fillna(-999)

    df_train.drop(columns=['Transaction_Day'], inplace=True)

    # --- 5. RISK FRAUD FEATURE ENGINEERING ---
    df_train['Hour'] = (df_train['TransactionDT'] // 3600) % 24
    df_train['DayOfWeek'] = (df_train['TransactionDT'] // 86400) % 7

    def assign_hour_risk(hour: int) -> str:
        if hour in [2, 3, 4, 5]: return 'high'
        if hour in [0, 1, 6, 23]: return 'medium'
        if hour in [7, 8, 9, 10, 11, 12]: return 'low'
        return 'very_low'

    df_train['hour_fraud_status'] = df_train['Hour'].apply(assign_hour_risk)
    df_train.drop(columns=['TransactionDT'], inplace=True)

    # --- 6. USER ID (UID) AGGREGATIONS ---
    df_train['uid'] = df_train['card1'].astype(str) + '_' + df_train['card2'].astype(str) + '_' + df_train['addr1'].astype(str)
    uid_counts = df_train['uid'].value_counts()
    df_train['uid_count'] = df_train['uid'].map(uid_counts)

    uid_amt_mean = df_train.groupby('uid')['TransactionAmt'].mean()
    uid_amt_std = df_train.groupby('uid')['TransactionAmt'].std()

    df_train['uid_amt_mean'] = df_train['uid'].map(uid_amt_mean)
    df_train['uid_amt_std'] = df_train['uid'].map(uid_amt_std).fillna(0)
    df_train.drop(columns=['uid'], inplace=True)

    # --- 7. DEVICE & EMAIL ENGINEERING ---
    if 'DeviceInfo' in df_train.columns:
        df_train['DeviceInfo'] = df_train['DeviceInfo'].fillna('unknown_device').str.lower()
        
        def get_device_corp(device_string: str) -> str:
            if 'samsung' in device_string or 'sm-' in device_string: return 'samsung'
            if any(x in device_string for x in ['iphone', 'ipad', 'macos', 'apple']): return 'apple'
            if 'huawei' in device_string or 'ale-' in device_string: return 'huawei'
            if 'lg' in device_string or 'nexus' in device_string: return 'lg'
            if 'windows' in device_string or 'trident' in device_string: return 'microsoft'
            if 'unknown_device' in device_string: return 'unknown'
            return 'other_brands'

        df_train['Device_corp'] = df_train['DeviceInfo'].apply(get_device_corp)
        df_train.drop(columns=['DeviceInfo'], inplace=True)

    df_train['cent'] = df_train['TransactionAmt'] - np.floor(df_train['TransactionAmt'])
    df_train['LogTransactionAmt'] = np.log1p(df_train['TransactionAmt'])

    if 'P_emaildomain' in df_train.columns:
        df_train['P_emaildomain'] = df_train['P_emaildomain'].fillna('null')
        df_train['P_email_company'] = df_train['P_emaildomain'].apply(lambda x: x.split('.')[0])

    # --- 8. MISSING VALUES IMPUTATION ---
    categorical_cols = df_train.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        df_train[col] = df_train[col].fillna('null')

    for col in df_train.select_dtypes(include=np.number).columns:
        df_train[col] = df_train[col].fillna(-999)

    # --- 9. CATEGORICAL ENCODING & MEMORY OPTIMIZATION ---
    for col in categorical_cols:
        encoder = LabelEncoder()
        df_train[col] = df_train[col].astype(str)
        df_train[col] = encoder.fit_transform(df_train[col])
        df_train[col] = df_train[col].astype('int32') if df_train[col].max() > 32000 else df_train[col].astype('int16')

    # --- 10. FEAST FORMAT COMPLIANCE ALIGNMENT ---
    df_train = df_train.rename(columns={"TransactionID": "transaction_id"})
    df_train["transaction_id"] = df_train["transaction_id"].astype("int64")
    # df_train["created_timestamp"] = pd.Timestamp.now()
    df_train["created_timestamp"] = df_train["event_timestamp"]

    # Drop target variable for Feast storage
    feast_features = df_train.drop(columns=['isFraud'], errors='ignore')

    # Exporting into the designated Feast repository environment
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    feast_features.to_parquet(OUT_PARQUET, index=False)
    
    print(f"[seed] New file generated successfully. Wrote: {OUT_PARQUET} ({OUT_PARQUET.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()