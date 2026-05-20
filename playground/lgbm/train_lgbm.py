"""Train the LightGBM fraud predictor.

Extracted from 2025-05-01_Model.ipynb (cell 5 preprocessing + cell 10 training)
into a reproducible CLI. Produces:
  - lgbm_model.pkl       fitted LGBMClassifier
  - label_encoders.pkl   per-column LabelEncoders (so inference can encode
                         categorical inputs the same way training did)
  - metrics.json         AUC, hyperparams, row counts
  - feature_importance.csv

Usage:
    python train_lgbm.py \\
        --transaction-csv path/to/train_transaction.csv \\
        --identity-csv ../train_identity.csv \\
        --output-dir .                # default: same folder
        [--quick]                     # smoke test (n_estimators=200, no early stop)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

HERE = Path(__file__).resolve().parent

DROP_COLUMNS = [
    "id_07", "id_21", "id_22", "id_23", "id_24",
    "id_25", "id_26", "id_27", "TransactionID",
]
TXN_AMT_CAP = 20000
NULL_TOKEN = "null"
NUMERIC_FILLNA = -999.0

DEFAULT_HYPERPARAMS = {
    "learning_rate": 0.01,
    "max_depth": 12,
    "n_estimators": 10000,
    "bagging_fraction": 0.8,
    "feature_fraction": 0.4,
    "boosting_type": "gbdt",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--transaction-csv", type=Path, required=True,
                    help="Path to train_transaction.csv (~470 MB from Kaggle IEEE-CIS).")
    ap.add_argument("--identity-csv", type=Path,
                    default=HERE.parent / "train_identity.csv",
                    help="Path to train_identity.csv. Default: ../train_identity.csv")
    ap.add_argument("--output-dir", type=Path, default=HERE,
                    help="Where to write .pkl + metrics. Default: this folder.")
    ap.add_argument("--learning-rate", type=float, default=DEFAULT_HYPERPARAMS["learning_rate"])
    ap.add_argument("--max-depth", type=int, default=DEFAULT_HYPERPARAMS["max_depth"])
    ap.add_argument("--n-estimators", type=int, default=DEFAULT_HYPERPARAMS["n_estimators"])
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--early-stopping-rounds", type=int, default=50)
    ap.add_argument("--quick", action="store_true",
                    help="Lower n_estimators to 200 + no early stopping (smoke test, ~30s).")
    return ap.parse_args()


def load_and_merge(tx_path: Path, ident_path: Path) -> pd.DataFrame:
    print(f"[load] reading {tx_path}")
    tx = pd.read_csv(tx_path)
    print(f"[load] reading {ident_path}")
    ident = pd.read_csv(ident_path)
    df = tx.merge(ident, on="TransactionID", how="left")
    print(f"[load] merged shape: {df.shape}  fraud_rate: {df['isFraud'].mean():.4f}")
    return df


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, dict[str, LabelEncoder]]:
    # Drop high-missing / id columns.
    cols = [c for c in DROP_COLUMNS if c in df.columns]
    if cols:
        df = df.drop(columns=cols)

    # Drop outlier transactions.
    if "TransactionAmt" in df.columns:
        before = len(df)
        df = df[df["TransactionAmt"] <= TXN_AMT_CAP].reset_index(drop=True)
        print(f"[preprocess] dropped {before - len(df)} rows TransactionAmt > {TXN_AMT_CAP}")

    # Split y/X.
    y = df["isFraud"].astype(int)
    X = df.drop(columns=["isFraud"])

    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=np.number).columns.tolist()

    # Fillna.
    for c in cat_cols:
        X[c] = X[c].fillna(NULL_TOKEN).astype(str)
    for c in num_cols:
        X[c] = X[c].fillna(NUMERIC_FILLNA)

    # LabelEncode each categorical column.
    encoders: dict[str, LabelEncoder] = {}
    for c in cat_cols:
        le = LabelEncoder()
        X[c] = le.fit_transform(X[c])
        encoders[c] = le

    print(f"[preprocess] X: {X.shape}  categorical: {len(cat_cols)}  numeric: {len(num_cols)}")
    return X, y, encoders


def train(X: pd.DataFrame, y: pd.Series, *, hyperparams: dict, test_size: float,
          random_state: int, early_stopping_rounds: int, quick: bool) -> tuple[lgb.LGBMClassifier, dict]:
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state,
    )
    print(f"[train] train: {len(y_train)}  val: {len(y_val)}  fraud_rate_train: {y_train.mean():.4f}")

    if quick:
        hyperparams = dict(hyperparams)
        hyperparams["n_estimators"] = 200
        print("[train] QUICK mode: n_estimators=200, no early stopping")

    model = lgb.LGBMClassifier(random_state=random_state, **hyperparams)

    fit_kwargs = {"eval_set": [(X_val, y_val)]}
    if not quick and early_stopping_rounds > 0:
        fit_kwargs["callbacks"] = [lgb.early_stopping(early_stopping_rounds, verbose=False)]

    t0 = time.time()
    model.fit(X_train, y_train, **fit_kwargs)
    seconds = time.time() - t0
    print(f"[train] done in {seconds:.1f}s  best_iter: {getattr(model, 'best_iteration_', None)}")

    # Evaluate.
    proba = model.predict_proba(X_val)[:, 1]
    pred = (proba >= 0.5).astype(int)
    auc = float(roc_auc_score(y_val, proba))
    cm = confusion_matrix(y_val, pred).tolist()
    report = classification_report(y_val, pred, output_dict=True, zero_division=0)
    print(f"[train] val AUC: {auc:.4f}")

    metrics = {
        "auc": round(auc, 4),
        "train_seconds": round(seconds, 2),
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "fraud_rate_train": round(float(y_train.mean()), 4),
        "fraud_rate_val": round(float(y_val.mean()), 4),
        "hyperparams": hyperparams,
        "best_iteration": getattr(model, "best_iteration_", None),
        "confusion_matrix": cm,
        "classification_report": report,
    }
    return model, metrics


def save_artifacts(model: lgb.LGBMClassifier, encoders: dict, metrics: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)

    model_path = out / "lgbm_model.pkl"
    joblib.dump(model, model_path)
    print(f"[save] {model_path} ({model_path.stat().st_size / 1024 / 1024:.1f} MB)")

    enc_path = out / "label_encoders.pkl"
    joblib.dump(encoders, enc_path)
    print(f"[save] {enc_path} ({enc_path.stat().st_size / 1024:.1f} KB)")

    metrics_path = out / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, default=str))
    print(f"[save] {metrics_path}")

    imp = pd.DataFrame({
        "feature": model.feature_name_,
        "gain": model.booster_.feature_importance(importance_type="gain"),
        "split": model.booster_.feature_importance(importance_type="split"),
    })
    imp["gain_pct"] = (imp["gain"] / imp["gain"].sum() * 100).round(2)
    imp = imp.sort_values("gain", ascending=False).reset_index(drop=True)
    imp.insert(0, "rank", imp.index + 1)
    imp_path = out / "feature_importance.csv"
    imp.to_csv(imp_path, index=False)
    print(f"[save] {imp_path} ({len(imp)} features)")


def main() -> int:
    args = parse_args()

    if not args.transaction_csv.exists():
        print(f"ERROR: transaction-csv not found: {args.transaction_csv}", file=sys.stderr)
        print("       Download train_transaction.csv from "
              "https://www.kaggle.com/c/ieee-fraud-detection/data", file=sys.stderr)
        return 1
    if not args.identity_csv.exists():
        print(f"ERROR: identity-csv not found: {args.identity_csv}", file=sys.stderr)
        return 1

    df = load_and_merge(args.transaction_csv, args.identity_csv)
    X, y, encoders = preprocess(df)
    hyperparams = {
        "learning_rate": args.learning_rate,
        "max_depth": args.max_depth,
        "n_estimators": args.n_estimators,
        "bagging_fraction": 0.8,
        "feature_fraction": 0.4,
        "boosting_type": "gbdt",
    }
    model, metrics = train(
        X, y,
        hyperparams=hyperparams,
        test_size=args.test_size,
        random_state=args.random_state,
        early_stopping_rounds=args.early_stopping_rounds,
        quick=args.quick,
    )
    save_artifacts(model, encoders, metrics, args.output_dir)
    print(f"\n[done] AUC={metrics['auc']}  output: {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
