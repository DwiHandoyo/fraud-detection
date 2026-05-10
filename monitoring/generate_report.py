"""Generate an Evidently HTML drift report.

Splits train_identity.csv (or feature-store/data/identity.parquet) into two
halves by row order — earlier half = reference, later half = current. Runs
Evidently DataDriftPreset and writes report to reports/drift_report.html.

In a real production setup, the "current" dataset would be the last 7 days of
prediction logs. Splitting the static IEEE-CIS file is a stand-in until
ml-service produces enough live traffic.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
PARQUET = PROJECT / "feature-store" / "feature_repo" / "data" / "identity.parquet"
CSV = PROJECT / "playground" / "train_identity.csv"
OUT_DIR = HERE / "reports"

# Columns to monitor — top features the model actually uses, mapping LGBM/EBM importance.
NUMERIC_COLS = ["id_01", "id_02", "id_05", "id_11", "id_13", "id_14", "id_17", "id_19"]
CATEGORICAL_COLS = ["id_15", "id_30", "id_31", "id_33", "id_35", "DeviceType", "DeviceInfo"]


def load() -> pd.DataFrame:
    if PARQUET.exists():
        print(f"[load] {PARQUET}")
        return pd.read_parquet(PARQUET)
    if CSV.exists():
        print(f"[load] {CSV}")
        return pd.read_csv(CSV)
    raise FileNotFoundError("No source data. Run feature-store/seed_data.py first.")


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    df = load()
    cols = [c for c in NUMERIC_COLS + CATEGORICAL_COLS if c in df.columns]
    df = df[cols].copy()

    midpoint = len(df) // 2
    reference = df.iloc[:midpoint].reset_index(drop=True)
    current = df.iloc[midpoint:].reset_index(drop=True)
    print(f"[split] reference: {len(reference)} rows | current: {len(current)} rows")
    print(f"[split] columns monitored ({len(cols)}): {cols}")

    report = Report(metrics=[
        DataDriftPreset(),
        DataSummaryPreset(),
    ])
    snapshot = report.run(reference_data=reference, current_data=current)

    html_path = OUT_DIR / "drift_report.html"
    snapshot.save_html(str(html_path))
    print(f"[done] wrote {html_path} ({html_path.stat().st_size // 1024} KB)")

    json_path = OUT_DIR / "drift_summary.json"
    snapshot.save_json(str(json_path))
    print(f"[done] wrote {json_path}")

    # Print quick CLI summary.
    snap_dict = snapshot.dict()
    drifted = 0
    for metric in snap_dict.get("metrics", []):
        mid = metric.get("metric_id", "")
        if "DriftedColumnsCount" in mid:
            drifted = int(metric.get("value", {}).get("count", 0))
            break
    print(f"[summary] drifted columns: {drifted} / {len(cols)}")


if __name__ == "__main__":
    main()
