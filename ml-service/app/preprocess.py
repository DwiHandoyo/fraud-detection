"""
Reusable preprocessing pipeline for the IEEE-CIS fraud detection dataset.

Encapsulates the steps from 2025-03-15_Preprocessing.ipynb so they can be
fit once on training data, saved to disk, and applied to new rows
deterministically.

Pipeline order:
    1. Rename test columns: 'id-XX' -> 'id_XX'
    2. Drop high-missing / id columns
    3. Drop outliers: TransactionAmt > 20000
    4. Fillna: 'null' for categorical, -999 for numeric
    5. LabelEncoder for each categorical column (unseen categories -> 'null')

Usage:
    pre = Preprocessor()
    X_train = pre.fit_transform(train_df)   # fit on training data
    pre.save("preprocessor.pkl")            # persist fitted state
    # later, on new data:
    pre2 = Preprocessor.load("preprocessor.pkl")
    X_new = pre2.transform(new_df)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

DEFAULT_DROP_COLUMNS = [
    "id_07", "id_21", "id_22", "id_23", "id_24",
    "id_25", "id_26", "id_27", "TransactionID",
]
TXN_AMT_CAP = 20000
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
    feature_names: list[str] = field(default_factory=list)
    fitted: bool = False

    @staticmethod
    def _rename_test_id_cols(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = df.columns.str.replace(r"^id-", "id_", regex=True)
        return df

    def _drop(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in self.drop_columns if c in df.columns]
        if cols:
            df = df.drop(columns=cols)
        return df

    def _drop_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        if "TransactionAmt" in df.columns:
            df = df[df["TransactionAmt"] <= self.txn_amt_cap].reset_index(drop=True)
        return df

    def _fillna(self, df: pd.DataFrame, cat_cols: list[str], num_cols: list[str]) -> pd.DataFrame:
        for col in cat_cols:
            df[col] = df[col].fillna(self.null_token).astype(str)
        for col in num_cols:
            df[col] = df[col].fillna(self.numeric_fillna)
        return df

    def fit(self, df: pd.DataFrame, drop_outliers: bool = True) -> "Preprocessor":
        df = self._rename_test_id_cols(df)
        df = self._drop(df)
        if drop_outliers:
            df = self._drop_outliers(df)

        self.categorical_cols = list(df.select_dtypes(include=["object", "category"]).columns)
        self.numeric_cols = list(df.select_dtypes(include=np.number).columns)

        df = self._fillna(df.copy(), self.categorical_cols, self.numeric_cols)

        self.encoders = {}
        for col in self.categorical_cols:
            le = LabelEncoder()
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
        df = self._drop(df)
        if drop_outliers:
            df = self._drop_outliers(df)

        cat_present = [c for c in self.categorical_cols if c in df.columns]
        num_present = [c for c in self.numeric_cols if c in df.columns]
        df = self._fillna(df.copy(), cat_present, num_present)

        for col in cat_present:
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
