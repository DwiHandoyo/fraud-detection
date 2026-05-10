"""Convert train_identity.csv to a Feast-ready parquet file.

Adds:
  - transaction_id (int64) — copied from TransactionID, used as the entity join key
  - event_timestamp (datetime) — synthetic, derived from a reference date.
    train_identity.csv has no timestamp column. We reuse the row order as the
    timestamp ordering so Feast point-in-time joins behave reasonably.

Output: feature_repo/data/identity.parquet
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
PLAYGROUND = HERE.parent / "playground"
SOURCE_CSV = PLAYGROUND / "train_identity.csv"
OUT_PARQUET = HERE / "feature_repo" / "data" / "identity.parquet"

# Reference date used by the IEEE-CIS Kaggle community (rough convention).
REFERENCE_DATE = datetime(2017, 12, 1)


def main() -> None:
    print(f"[seed] reading {SOURCE_CSV}")
    df = pd.read_csv(SOURCE_CSV)
    print(f"[seed] loaded {len(df)} rows, {len(df.columns)} cols")

    df = df.rename(columns={"TransactionID": "transaction_id"})
    df["transaction_id"] = df["transaction_id"].astype("int64")

    # Synthetic monotonic timestamps spaced 1 second apart, anchored at REFERENCE_DATE.
    df["event_timestamp"] = pd.to_datetime(
        [REFERENCE_DATE + timedelta(seconds=i) for i in range(len(df))]
    )
    # created_timestamp is required by Feast for offline → online materialization.
    df["created_timestamp"] = df["event_timestamp"]

    # Coerce numeric columns to float32 (smaller online store, matches schema).
    for col in df.columns:
        if col in ("transaction_id", "event_timestamp", "created_timestamp"):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].astype("float32")

    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PARQUET, index=False)
    print(f"[seed] wrote {OUT_PARQUET} ({OUT_PARQUET.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"[seed] columns: {list(df.columns)[:8]} ... ({len(df.columns)} total)")
    print(f"[seed] sample row:")
    print(df.iloc[0].to_dict())


if __name__ == "__main__":
    main()
