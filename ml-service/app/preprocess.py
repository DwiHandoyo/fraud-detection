"""
Reusable preprocessing and feature engineering pipeline for the IEEE-CIS fraud detection dataset.

Encapsulates advanced feature engineering logic (UID aggregations, Device brand parsing, 
and correlation-based V-feature reduction) so they can be fit once on training data, 
saved to disk, and applied to new rows deterministically during inference.

Pipeline order:
    1. Standardize test columns by renaming 'id-XX' to 'id_XX'.
    2. Identify and drop redundant, highly correlated V-feature structural groups.
    3. Drop sparse identity columns (id_07, id_08, id_21-id_27) and extreme nominal 
       outliers (TransactionAmt > 30000).
    4. Apply Time-Normalized Delta conversion to D-features (D1-D15) relative to TransactionDT.
    5. Engineer operational time features (Hour, DayOfWeek, hour_fraud_status).
    6. Construct User Unique Identifiers (UID) using card and address mappings to 
       calculate safe historical aggregates (uid_count, uid_amt_mean, uid_amt_std).
    7. Segment DeviceInfo into parent brand corporations (Device_corp) and execute 
       mathematical transformations (cent, LogTransactionAmt).
    8. Extract P_emaildomain into standalone company features (P_email_company).
    9. Handle missing values: Fill 'null' for categorical, -999 for numeric variables.
    10. Encode all categorical columns via LabelEncoder (unseen test categories -> 'null')
        and optimize memory allocations to int16/int32.

Usage:
    pre = Preprocessor()
    X_train = pre.fit_transform(train_df)    # fit on training data & compute UID stats
    pre.save("preprocessor.pkl")             # persist fitted encoders & stats state
    
    # later, on new inference streams or test data:
    pre2 = Preprocessor.load("preprocessor.pkl")
    X_new = pre2.transform(new_df)           # transforms new rows deterministically
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

DEFAULT_DROP_COLUMNS = [
    "id_07", "id_08", "id_21", "id_22", "id_23", "id_24",
    "id_25", "id_26", "id_27", "TransactionID",
]
TXN_AMT_CAP = 30000  # Matched with your outlier filtering threshold
NULL_TOKEN = "null"
NUMERIC_FILLNA = -999


@dataclass
class Preprocessor:
    drop_columns: list[str] = field(default_factory=lambda: list(DEFAULT_DROP_COLUMNS))
    txn_amt_cap: float = TXN_AMT_CAP
    null_token: str = NULL_TOKEN
    numeric_fillna: float = NUMERIC_FILLNA

    encoders: dict[str, LabelEncoder] = field(default_factory=dict)
    categorical_cols: list[str] = field(default_factory=list)
    numeric_cols: list[str] = field(default_factory=list)
    v_cols_to_drop: list[str] = field(default_factory=list)
    
    # Storage for advanced user entity statistics calculated during training fit
    uid_counts: pd.Series = field(default_factory=lambda: pd.Series(dtype="float32"))
    uid_amt_mean: pd.Series = field(default_factory=lambda: pd.Series(dtype="float32"))
    uid_amt_std: pd.Series = field(default_factory=lambda: pd.Series(dtype="float32"))
    
    feature_names: list[str] = field(default_factory=list)
    fitted: bool = False

    @staticmethod
    def _rename_test_id_cols(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = df.columns.str.replace(r"^id-", "id_", regex=True)
        return df

    def _drop_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        if "TransactionAmt" in df.columns:
            df = df[df["TransactionAmt"] <= self.txn_amt_cap].reset_index(drop=True)
        return df

    def _reduce_v_groups(self, df: pd.DataFrame) -> list[str]:
        """Finds highly correlated structural V groups with missing values to drop."""
        v_missing_cols = [col for col in df.columns if col.startswith('V') and df[col].isnull().any()]
        v_nan_dict = {}
        for col in v_missing_cols:
            count = df[col].isnull().sum()
            v_nan_dict.setdefault(count, []).append(col)

        v_groups_to_reduce = [v for k, v in v_nan_dict.items() if len(v) > 1]
        all_v_in_groups = [col for group in v_groups_to_reduce for col in group]
        
        # Keep features with max unique values inside each subset group
        use = []
        for col_group in v_groups_to_reduce:
            max_unique, max_index = 0, 0
            for i, c in enumerate(col_group):
                n = df[c].nunique()
                if n > max_unique:
                    max_unique, max_index = n, i
            use.append(col_group[max_index])
            
        return list(set(all_v_in_groups) - set(use))

    def _apply_feature_engineering(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """Applies custom complex domain-specific features safely across execution modes."""
        df = df.copy()
        
        # 1. D-Features Time-Normalized Delta conversion
        if 'TransactionDT' in df.columns:
            df['Transaction_Day'] = df['TransactionDT'] / 86400
            for i in range(1, 16):
                col = f'D{i}'
                if col in df.columns:
                    df[f'{col}_norm'] = df[col] - df['Transaction_Day']
                    df.drop(columns=[col], inplace=True)
                    df[f'{col}_norm'] = df[f'{col}_norm'].fillna(self.numeric_fillna)
            df.drop(columns=['Transaction_Day'], inplace=True)

        # 2. Hour Fraud Risk status extraction
        if 'TransactionDT' in df.columns:
            df['Hour'] = (df['TransactionDT'] // 3600) % 24
            df['DayOfWeek'] = (df['TransactionDT'] // 86400) % 7
            
            def assign_hour_risk(hour: int) -> str:
                if hour in [2, 3, 4, 5]: return 'high'
                if hour in [0, 1, 6, 23]: return 'medium'
                if hour in [7, 8, 9, 10, 11, 12]: return 'low'
                return 'very_low'
                
            df['hour_fraud_status'] = df['Hour'].apply(assign_hour_risk)
            df.drop(columns=['TransactionDT'], inplace=True, errors='ignore')

        # 3. User Unique Identifier (UID) mapping
        if all(c in df.columns for c in ['card1', 'card2', 'addr1']):
            df['uid'] = df['card1'].astype(str) + '_' + df['card2'].astype(str) + '_' + df['addr1'].astype(str)
            
            if is_training:
                self.uid_counts = df['uid'].value_counts()
                self.uid_amt_mean = df.groupby('uid')['TransactionAmt'].mean()
                self.uid_amt_std = df.groupby('uid')['TransactionAmt'].std().fillna(0)

            df['uid_count'] = df['uid'].map(self.uid_counts).fillna(1)
            df['uid_amt_mean'] = df['uid'].map(self.uid_amt_mean).fillna(df['TransactionAmt'])
            df['uid_amt_std'] = df['uid'].map(self.uid_amt_std).fillna(0)
            df.drop(columns=['uid'], inplace=True)

        # 4. Device Brand segmentation and Cent/Log transformation
        if 'DeviceInfo' in df.columns:
            df['DeviceInfo'] = df['DeviceInfo'].fillna('unknown_device').str.lower()
            def get_device_corp(ds: str) -> str:
                if 'samsung' in ds or 'sm-' in ds: return 'samsung'
                if any(x in ds for x in ['iphone', 'ipad', 'macos', 'apple']): return 'apple'
                if 'huawei' in ds or 'ale-' in ds: return 'huawei'
                if 'lg' in ds or 'nexus' in ds: return 'lg'
                if 'windows' in ds or 'trident' in ds: return 'microsoft'
                return 'unknown' if 'unknown_device' in ds else 'other_brands'
            df['Device_corp'] = df['DeviceInfo'].apply(get_device_corp)
            df.drop(columns=['DeviceInfo'], inplace=True)

        if 'TransactionAmt' in df.columns:
            df['cent'] = df['TransactionAmt'] - np.floor(df['TransactionAmt'])
            df['LogTransactionAmt'] = np.log1p(df['TransactionAmt'])

        if 'P_emaildomain' in df.columns:
            df['P_emaildomain'] = df['P_emaildomain'].fillna('null')
            df['P_email_company'] = df['P_emaildomain'].apply(lambda x: x.split('.')[0])

        return df

    def fit(self, df: pd.DataFrame, drop_outliers: bool = True) -> "Preprocessor":
        df = self._rename_test_id_cols(df)
        
        # Identify redundant V cols from current fit stream
        self.v_cols_to_drop = self._reduce_v_groups(df)
        df = df.drop(columns=self.v_cols_to_drop, errors='ignore')
        
        if drop_outliers:
            df = self._drop_outliers(df)
            
        # Drop duplicates at training level
        duplicate_cols = df.columns.drop('TransactionID')
        df = df.drop_duplicates(subset=duplicate_cols, keep='first').reset_index(drop=True)

        # Build structural features
        df = self._apply_feature_engineering(df, is_training=True)
        df = df.drop(columns=[c for c in self.drop_columns if c in df.columns], errors='ignore')

        self.categorical_cols = list(df.select_dtypes(include=["object", "category"]).columns)
        self.numeric_cols = list(df.select_dtypes(include=np.number).columns)

        # Build Label Encoders
        for col in self.categorical_cols:
            le = LabelEncoder()
            df[col] = df[col].fillna(self.null_token).astype(str)
            uniq = pd.concat([df[col], pd.Series([self.null_token])]).unique()
            le.fit(uniq)
            self.encoders[col] = le

        self.feature_names = [c for c in df.columns if c != "isFraud"]
        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame, drop_outliers: bool = False) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError("Preprocessor must be fit() before transform()")

        df = self._rename_test_id_cols(df)
        df = df.drop(columns=self.v_cols_to_drop, errors='ignore')
        
        if drop_outliers:
            df = self._drop_outliers(df)

        df = self._apply_feature_engineering(df, is_training=False)
        df = df.drop(columns=[c for c in self.drop_columns if c in df.columns], errors='ignore')

        # Fillna and apply Label Encoding mapping securely
        cat_present = [c for c in self.categorical_cols if c in df.columns]
        num_present = [c for c in self.numeric_cols if c in df.columns]

        for col in num_present:
            df[col] = df[col].fillna(self.numeric_fillna)

        for col in cat_present:
            df[col] = df[col].fillna(self.null_token).astype(str)
            le = self.encoders[col]
            known = set(le.classes_)
            df[col] = df[col].where(df[col].isin(known), self.null_token)
            df[col] = le.transform(df[col].astype(str))
            df[col] = df[col].astype("int32" if df[col].max() > 32000 else "int16")

        return df

    def fit_transform(self, df: pd.DataFrame, drop_outliers: bool = True) -> pd.DataFrame:
        return self.fit(df, drop_outliers=drop_outliers).transform(df, drop_outliers=drop_outliers)

    def save(self, path: str | Path) -> None:
        joblib.dump(self, Path(path))

    @classmethod
    def load(cls, path: str | Path) -> "Preprocessor":
        obj = joblib.load(Path(path))
        if not isinstance(obj, cls):
            raise TypeError(f"Loaded object is not Preprocessor: {type(obj)}")
        return obj