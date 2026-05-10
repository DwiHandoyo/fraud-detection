"""
Fit the Preprocessor on available training data and save to preprocessor.pkl.

Modes:
  --identity-only: fit on train_identity.csv (works out-of-the-box)
  --full PATH    : fit on full merged data (requires train_transaction.csv)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from preprocess import Preprocessor

HERE = Path(__file__).parent
PLAYGROUND = HERE.parent
IDENTITY_CSV = PLAYGROUND / "train_identity.csv"
OUT_PATH = HERE / "preprocessor.pkl"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", type=Path, default=None,
                    help="Path to train_transaction.csv for full pipeline.")
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    args = ap.parse_args()

    if args.full and args.full.exists():
        print(f"[mode] full pipeline: {args.full}")
        tx = pd.read_csv(args.full)
        ident = pd.read_csv(IDENTITY_CSV)
        df = tx.merge(ident, on="TransactionID", how="left")
        print(f"[data] merged shape: {df.shape}")
    else:
        print(f"[mode] identity-only: {IDENTITY_CSV}")
        df = pd.read_csv(IDENTITY_CSV)
        print(f"[data] shape: {df.shape}")

    pre = Preprocessor()
    print(f"[fit] dropping: {pre.drop_columns}")
    df_proc = pre.fit_transform(df)

    print(f"[done] processed shape: {df_proc.shape}")
    print(f"[done] categorical cols ({len(pre.categorical_cols)}): {pre.categorical_cols[:8]} ...")
    print(f"[done] numeric cols ({len(pre.numeric_cols)}): {pre.numeric_cols[:8]} ...")
    print(f"[done] sample row:")
    print(df_proc.head(2).to_string())

    pre.save(args.out)
    print(f"\n[done] saved fitted preprocessor to {args.out}")
    print(f"[done] file size: {args.out.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
