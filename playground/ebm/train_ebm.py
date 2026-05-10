"""
Train an Explainable Boosting Machine (EBM) as a transparent surrogate
for the existing LightGBM fraud model.

Default mode is knowledge distillation: the LightGBM teacher labels rows from
train_identity.csv, then the EBM student is trained on a feature subset.
Pass --transaction-csv to switch to supervised mode using real isFraud labels.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

HERE = Path(__file__).parent
PLAYGROUND = HERE.parent
LGBM_PATH = PLAYGROUND / "lgbm_tuning.pkl"
IDENTITY_CSV = PLAYGROUND / "train_identity.csv"
FEATURE_IMP_CSV = PLAYGROUND / "feature_importance.csv"

MODEL_OUT = HERE / "ebm_model.pkl"
METRICS_OUT = HERE / "metrics.json"
TOP_N_FEATURES = 25


def load_top_features(n: int, restrict_to: set[str] | None = None) -> list[str]:
    """Return top-n features by LGBM gain. If restrict_to is given, keep only
    those features that intersect (used for distillation mode where the
    student can only see columns available in train_identity.csv)."""
    fi = pd.read_csv(FEATURE_IMP_CSV)
    feats = fi["feature"].tolist()
    if restrict_to is not None:
        feats = [f for f in feats if f in restrict_to]
    return feats[:n]


def build_distillation_dataset(
    teacher, expected_features: list[str], top_features: list[str]
) -> tuple[pd.DataFrame, np.ndarray]:
    """Use train_identity.csv rows + LightGBM predictions as soft labels."""
    print(f"[data] reading {IDENTITY_CSV.name} ...")
    identity = pd.read_csv(IDENTITY_CSV)
    print(f"[data] identity rows: {len(identity)}")

    # Build full feature matrix expected by teacher (zero-fill missing).
    full = pd.DataFrame(0.0, index=identity.index, columns=expected_features)
    for col in expected_features:
        if col in identity.columns:
            full[col] = identity[col]

    obj_cols = full.select_dtypes(include=["object"]).columns
    for c in obj_cols:
        full[c] = pd.factorize(full[c])[0]
    full = full.fillna(0.0)

    print("[teacher] generating soft labels from LightGBM ...")
    proba = teacher.predict_proba(full)[:, 1]
    print(
        f"[teacher] proba stats: min={proba.min():.4f} "
        f"max={proba.max():.4f} mean={proba.mean():.4f}"
    )

    # Keep raw values (object + numeric) for EBM — it handles both natively.
    student_X = identity.reindex(columns=top_features).copy()
    return student_X, proba


def build_supervised_dataset(
    transaction_csv: Path, top_features: list[str]
) -> tuple[pd.DataFrame, np.ndarray]:
    print(f"[data] reading {transaction_csv} ...")
    tx = pd.read_csv(transaction_csv)
    print(f"[data] reading {IDENTITY_CSV.name} ...")
    ident = pd.read_csv(IDENTITY_CSV)
    df = tx.merge(ident, on="TransactionID", how="left")
    print(f"[data] merged rows: {len(df)} | fraud rate: {df['isFraud'].mean():.4f}")

    y = df["isFraud"].astype(int).to_numpy()
    X = df.reindex(columns=top_features).copy()
    return X, y


def train(
    X: pd.DataFrame, y: np.ndarray, soft_labels: bool
) -> tuple[ExplainableBoostingClassifier, dict]:
    # Distillation: use median split so the student learns the teacher's
    # *ranking* (calibration is biased anyway because most teacher inputs
    # are zero-filled outside the identity feature space).
    if soft_labels:
        threshold = float(np.median(y))
        y_binary = (y >= threshold).astype(int)
        print(f"[train] distillation threshold (median): {threshold:.4f}")
    else:
        y_binary = y.astype(int)

    print(
        f"[train] X shape: {X.shape} | "
        f"positive rate: {y_binary.mean():.4f}"
    )

    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
    )

    ebm = ExplainableBoostingClassifier(
        interactions=8,
        learning_rate=0.02,
        max_bins=256,
        max_interaction_bins=32,
        outer_bags=4,
        random_state=42,
        n_jobs=-1,
    )

    t0 = time.time()
    ebm.fit(X_tr, y_tr)
    train_seconds = time.time() - t0

    val_proba = ebm.predict_proba(X_va)[:, 1]
    val_auc = float(roc_auc_score(y_va, val_proba))

    metrics = {
        "mode": "distillation" if soft_labels else "supervised",
        "n_features": X.shape[1],
        "n_train": int(len(y_tr)),
        "n_val": int(len(y_va)),
        "train_seconds": round(train_seconds, 2),
        "val_auc": round(val_auc, 4),
        "positive_rate": round(float(y_binary.mean()), 4),
    }

    if soft_labels:
        # Fidelity: how well student probabilities track teacher probabilities.
        student_full = ebm.predict_proba(X)[:, 1]
        teacher_full = y  # original soft labels
        fidelity_corr = float(np.corrcoef(student_full, teacher_full)[0, 1])
        fidelity_auc = float(
            roc_auc_score((teacher_full >= 0.5).astype(int), student_full)
        )
        metrics["fidelity_pearson"] = round(fidelity_corr, 4)
        metrics["fidelity_auc"] = round(fidelity_auc, 4)

    return ebm, metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--transaction-csv",
        type=Path,
        default=None,
        help="Optional path to train_transaction.csv for supervised mode.",
    )
    ap.add_argument(
        "--top-n",
        type=int,
        default=TOP_N_FEATURES,
        help=f"Number of top LGBM features to use (default {TOP_N_FEATURES}).",
    )
    args = ap.parse_args()

    print(f"[setup] loading teacher: {LGBM_PATH.name}")
    teacher = joblib.load(LGBM_PATH)
    expected_features = list(teacher.feature_name_)

    use_supervised = args.transaction_csv and args.transaction_csv.exists()
    if args.transaction_csv and not use_supervised:
        print(
            f"[warn] {args.transaction_csv} not found — "
            "falling back to distillation mode."
        )

    if use_supervised:
        top_features = load_top_features(args.top_n)
        print(f"[setup] top {len(top_features)} features: {top_features[:8]} ...")
        X, y = build_supervised_dataset(args.transaction_csv, top_features)
        soft = False
    else:
        # Distillation: student sees only features present in train_identity.csv.
        identity_cols = set(pd.read_csv(IDENTITY_CSV, nrows=1).columns)
        top_features = load_top_features(args.top_n, restrict_to=identity_cols)
        print(
            f"[setup] distillation mode — restricting student to {len(top_features)} "
            f"identity features: {top_features[:8]} ..."
        )
        X, y = build_distillation_dataset(teacher, expected_features, top_features)
        soft = True

    ebm, metrics = train(X, y, soft_labels=soft)

    joblib.dump({"model": ebm, "feature_names": list(X.columns)}, MODEL_OUT)
    METRICS_OUT.write_text(json.dumps(metrics, indent=2))

    print(f"\n[done] saved {MODEL_OUT.name}")
    print(f"[done] saved {METRICS_OUT.name}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
