"""Retrain LGBM from reviewer-corrected labels in bulk_predictions.

Pipeline:
  1. Query Postgres: rows with user_label in ('fraud','legit')
       - NEW: labeled within `--recent-days` window  (100% used)
       - OLD: older labels                            (sampled at --history-sample-rate)
  2. Reconstruct DataFrame from request_payload JSONB + user_label as target
  3. Reuse train_lgbm.preprocess + train (no duplication)
  4. Save versioned pkl + write training_jobs row in Postgres
  5. Print promotion command (manual promote step)

Usage:
    python retrain_from_feedback.py \
        --recent-days 30 \
        --history-sample-rate 0.20 \
        --min-new-labels 50 \
        --output-dir ../../ml-service/models \
        [--quick]

Env:
    DATABASE_URL=postgresql+psycopg2://fraud:fraud_demo_only@localhost:5432/fraud
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import joblib
import pandas as pd
from sqlalchemy import create_engine, text

# Reuse training logic from train_lgbm.py
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_lgbm import DEFAULT_HYPERPARAMS, preprocess, save_artifacts, train  # noqa: E402

HERE = Path(__file__).resolve().parent
DEFAULT_DB_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://fraud:fraud_demo_only@localhost:5432/fraud",
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--db-url", default=DEFAULT_DB_URL,
                    help="Postgres connection string.")
    ap.add_argument("--recent-days", type=int, default=30,
                    help="Window: labels within last N days = NEW (100%% used).")
    ap.add_argument("--history-sample-rate", type=float, default=0.20,
                    help="Sample rate for labels older than recent-days. 0.20 = 20%%.")
    ap.add_argument("--min-new-labels", type=int, default=20,
                    help="Refuse to retrain if fewer than N new labeled rows available.")
    ap.add_argument("--output-dir", type=Path,
                    default=Path(__file__).resolve().parent.parent.parent
                    / "ml-service" / "models",
                    help="Where to write the new versioned model pkl.")
    ap.add_argument("--quick", action="store_true",
                    help="Quick training (n_estimators=200, no early stop).")
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--notes", type=str, default="",
                    help="Optional note saved to training_jobs.notes")
    return ap.parse_args()


def fetch_labeled_rows(engine, recent_days: int, history_sample_rate: float) -> tuple[pd.DataFrame, int, int]:
    """Return concatenated DataFrame of NEW + OLD labeled rows."""
    cutoff_sql = f"NOW() - INTERVAL '{int(recent_days)} days'"

    with engine.connect() as conn:
        # NEW window
        new_rows = conn.execute(text(f"""
            SELECT request_payload, user_label, labeled_at
            FROM bulk_predictions
            WHERE user_label IN ('fraud', 'legit')
              AND labeled_at IS NOT NULL
              AND labeled_at >= {cutoff_sql}
        """)).mappings().all()

        # OLD: stratified random sample. Postgres has TABLESAMPLE but it's
        # block-level; use random() for row-level proper sample on small tables.
        old_rows = conn.execute(text(f"""
            SELECT request_payload, user_label, labeled_at
            FROM bulk_predictions
            WHERE user_label IN ('fraud', 'legit')
              AND labeled_at IS NOT NULL
              AND labeled_at < {cutoff_sql}
              AND random() < :sample_rate
        """), {"sample_rate": float(history_sample_rate)}).mappings().all()

    def _expand(rows: list[dict]) -> pd.DataFrame:
        out = []
        for r in rows:
            payload = r["request_payload"] or {}
            payload = dict(payload)  # copy
            payload["isFraud"] = 1 if r["user_label"] == "fraud" else 0
            out.append(payload)
        return pd.DataFrame(out) if out else pd.DataFrame()

    df_new = _expand(new_rows)
    df_old = _expand(old_rows)
    n_new, n_old = len(df_new), len(df_old)

    df = pd.concat([df_new, df_old], ignore_index=True) if not df_new.empty or not df_old.empty else pd.DataFrame()
    return df, n_new, n_old


def insert_training_job(engine, payload: dict) -> int | None:
    """Write a training_jobs row. Returns inserted id or None on error."""
    try:
        with engine.begin() as conn:
            result = conn.execute(
                text("""
                INSERT INTO training_jobs
                    (window_strategy, window_recent_days, window_history_sample,
                     train_cutoff_date, n_new_labels, n_old_labels, fraud_rate,
                     model_version, model_path, val_auc, train_seconds,
                     status, notes)
                VALUES
                    (:strategy, :recent, :sample, NOW() - INTERVAL ':recent days',
                     :n_new, :n_old, :fraud_rate,
                     :version, :path, :auc, :sec,
                     'trained', :notes)
                RETURNING id
                """),
                payload,
            )
            return int(result.scalar())
    except Exception as e:
        print(f"WARN: insert_training_job failed: {e}", file=sys.stderr)
        return None


def main() -> int:
    args = parse_args()
    engine = create_engine(args.db_url, pool_pre_ping=True)

    # 1. Fetch labeled rows.
    print(f"[fetch] window: last {args.recent_days} days = NEW, older = OLD ({args.history_sample_rate*100:.0f}% sample)")
    df, n_new, n_old = fetch_labeled_rows(engine, args.recent_days, args.history_sample_rate)

    if df.empty:
        print("ERROR: no labeled rows found. Label some bulk_predictions first via Bulk Audit page.",
              file=sys.stderr)
        return 2

    if n_new < args.min_new_labels:
        print(f"ERROR: only {n_new} new labels in last {args.recent_days} days "
              f"(min: {args.min_new_labels}). Skipping retrain.", file=sys.stderr)
        return 3

    print(f"[fetch] {n_new} NEW + {n_old} OLD = {len(df)} total rows")
    print(f"[fetch] fraud_rate: {df['isFraud'].mean():.4f}")

    # 2. Preprocess + train (reuse train_lgbm functions).
    X, y, encoders = preprocess(df)
    hyperparams = dict(DEFAULT_HYPERPARAMS)
    model, metrics = train(
        X, y,
        hyperparams=hyperparams,
        test_size=args.test_size,
        random_state=args.random_state,
        early_stopping_rounds=50,
        quick=args.quick,
    )

    # 3. Version + save.
    version = f"v{int(time.time())}"
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    model_path = out / f"lgbm_{version}.pkl"
    enc_path = out / f"label_encoders_{version}.pkl"

    joblib.dump(model, model_path)
    joblib.dump(encoders, enc_path)
    print(f"[save] {model_path} ({model_path.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"[save] {enc_path}")

    # Reuse train_lgbm.save_artifacts for metrics.json + feature_importance.csv
    # but write into a dedicated subdir to avoid clobbering champion artifacts.
    artifacts_dir = out / f"retrain_{version}"
    save_artifacts(model, encoders, metrics, artifacts_dir)

    # 4. Insert training_jobs row.
    job_id = insert_training_job(engine, {
        "strategy": f"sliding_{args.recent_days}d+history_{int(args.history_sample_rate*100)}pct",
        "recent": args.recent_days,
        "sample": args.history_sample_rate,
        "n_new": n_new,
        "n_old": n_old,
        "fraud_rate": float(df["isFraud"].mean()),
        "version": version,
        "path": str(model_path.relative_to(out.parent)),
        "auc": metrics["auc"],
        "sec": metrics["train_seconds"],
        "notes": args.notes or None,
    })

    print()
    print(f"[done] training_jobs.id = {job_id}")
    print(f"[done] val AUC = {metrics['auc']}")
    print()
    print(f"To promote this model, edit ml-service/config.yaml:")
    print(f"    predictor.path: models/lgbm_{version}.pkl")
    print(f"    predictor.version: \"{version}\"")
    print(f"Then restart: docker compose restart ml-service")
    print()
    print(f"Or use the UI: Continuous Training page → Promote button on job {job_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
