"""FairLearn audit — disparate impact analysis on LGBM predictions.

Memenuhi spec IF5251 nomor 5a (Responsible AI — fairness audit).

Caveat: train_transaction.csv tidak tersedia → tidak ada label isFraud asli.
Audit dilakukan pada *prediksi LGBM* (proxy label) untuk mendeteksi bias dalam
model (selection rate, demographic parity), bukan bias dataset (yang butuh
ground truth).

Fitur sensitif yang diaudit:
  - DeviceType: mobile vs desktop (proxy kelas ekonomi)
  - id_30: OS family (proxy device price tier)
  - id_31_family: browser family (chrome/safari/firefox/edge/other)

Output: reports/fairness_metrics.json + reports/fairness_summary.html
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    demographic_parity_ratio,
    selection_rate,
)

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
PARQUET = PROJECT / "feature-store" / "feature_repo" / "data" / "identity.parquet"
LGBM = PROJECT / "ml-service" / "models" / "lgbm.pkl"
OUT_DIR = HERE / "reports"

THRESHOLD = 0.5
SAMPLE_SIZE = 50_000


def os_family(value):
    if not isinstance(value, str):
        return "unknown"
    s = value.lower()
    if "windows" in s:
        return "Windows"
    if "ios" in s or "mac os" in s or "mac" in s:
        return "Apple"
    if "android" in s:
        return "Android"
    if "linux" in s:
        return "Linux"
    return "other"


def browser_family(value):
    if not isinstance(value, str):
        return "unknown"
    s = value.lower()
    for fam in ("chrome", "safari", "firefox", "edge", "ie", "samsung", "opera"):
        if fam in s:
            return fam
    return "other"


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    print(f"[load] {PARQUET}")
    df = pd.read_parquet(PARQUET)
    print(f"[load] {len(df)} rows")

    df = df.sample(SAMPLE_SIZE, random_state=42).reset_index(drop=True)
    print(f"[sample] {len(df)} rows for audit")

    print(f"[load] LGBM teacher from {LGBM}")
    model = joblib.load(LGBM)
    feats = list(model.feature_name_)

    # Prepare X aligned to LGBM feature schema, fill missing with 0.
    X = pd.DataFrame(0.0, index=df.index, columns=feats)
    for col in feats:
        if col in df.columns:
            X[col] = df[col]
    for c in X.select_dtypes(include=["object"]).columns:
        X[c] = pd.factorize(X[c])[0]
    X = X.fillna(0.0)

    print("[predict] generating LGBM probabilities ...")
    proba = model.predict_proba(X)[:, 1]
    y_pred = (proba >= THRESHOLD).astype(int)
    print(f"[predict] flagged {y_pred.sum()} / {len(y_pred)} ({y_pred.mean():.1%}) as fraud")

    # Sensitive features.
    sensitive = pd.DataFrame({
        "DeviceType": df["DeviceType"].fillna("unknown"),
        "OS_family": df["id_30"].apply(os_family),
        "Browser_family": df["id_31"].apply(browser_family),
    })

    # Compute metrics per sensitive group.
    metrics = {"selection_rate": selection_rate}
    results = {}
    for sensitive_col in sensitive.columns:
        groups = sensitive[sensitive_col]
        mf = MetricFrame(
            metrics=metrics,
            y_true=y_pred,  # we don't have ground truth; use predictions as both
            y_pred=y_pred,
            sensitive_features=groups,
        )
        dpd = demographic_parity_difference(y_pred, y_pred, sensitive_features=groups)
        dpr = demographic_parity_ratio(y_pred, y_pred, sensitive_features=groups)

        per_group = mf.by_group["selection_rate"].to_dict()
        per_group_count = groups.value_counts().to_dict()

        results[sensitive_col] = {
            "demographic_parity_difference": float(dpd),
            "demographic_parity_ratio": float(dpr),
            "selection_rate_overall": float(y_pred.mean()),
            "selection_rate_per_group": {str(k): float(v) for k, v in per_group.items()},
            "group_size": {str(k): int(v) for k, v in per_group_count.items()},
            "interpretation": _interpret(dpd, dpr),
        }

        print(f"\n[{sensitive_col}] DP_diff={dpd:.4f}  DP_ratio={dpr:.4f}")
        for grp, rate in sorted(per_group.items(), key=lambda x: -x[1]):
            n = per_group_count.get(grp, 0)
            print(f"   {str(grp):20s} n={n:6d} selection_rate={rate:.3f}")

    summary_path = OUT_DIR / "fairness_metrics.json"
    summary_path.write_text(json.dumps(results, indent=2))
    print(f"\n[done] wrote {summary_path}")

    _write_html(results, OUT_DIR / "fairness_summary.html")
    print(f"[done] wrote {OUT_DIR / 'fairness_summary.html'}")


def _interpret(dpd: float, dpr: float) -> str:
    """Plain-language interpretation of the parity scores."""
    if dpr >= 0.8 and abs(dpd) <= 0.10:
        return "PASS — within four-fifths rule (DPR>=0.80, |DPD|<=0.10)"
    if dpr >= 0.6:
        return "BORDERLINE — investigate group-level selection rates"
    return "FAIL — substantial disparate impact (DPR<0.60); review model"


def _write_html(results: dict, path: Path) -> None:
    rows_html = []
    for col, m in results.items():
        per_group = m["selection_rate_per_group"]
        sizes = m["group_size"]
        group_rows = "".join(
            f"<tr><td>{g}</td><td>{sizes.get(g,0):,}</td><td>{rate:.3f}</td></tr>"
            for g, rate in sorted(per_group.items(), key=lambda x: -x[1])
        )
        rows_html.append(f"""
        <h2>{col}</h2>
        <p><b>DP difference:</b> {m['demographic_parity_difference']:.4f} &nbsp;|&nbsp;
           <b>DP ratio:</b> {m['demographic_parity_ratio']:.4f} &nbsp;|&nbsp;
           <b>Verdict:</b> {m['interpretation']}</p>
        <table border="1" cellspacing="0" cellpadding="6">
          <tr><th>Group</th><th>Size</th><th>Selection rate</th></tr>
          {group_rows}
        </table>
        """)
    html = f"""<!DOCTYPE html><html><head>
    <meta charset="utf-8"><title>Fairness audit — fraud detection</title>
    <style>body{{font-family:system-ui,sans-serif;max-width:900px;margin:2em auto;padding:0 1em}}
    h1{{border-bottom:2px solid #2563eb;padding-bottom:0.3em}}
    h2{{margin-top:1.5em;color:#2563eb}}
    table{{border-collapse:collapse;margin-top:0.5em}}
    th{{background:#f3f4f6}}</style></head><body>
    <h1>Fairness audit — LGBM fraud model</h1>
    <p>Audit dilakukan pada {SAMPLE_SIZE:,} sampel dari train_identity.csv,
       menggunakan prediksi LGBM (threshold {THRESHOLD}) sebagai proxy label
       karena ground truth tidak tersedia.</p>
    <p>Metrik: <b>Demographic Parity Difference</b> (DPD; |max - min| selection rate
       antar group) dan <b>Demographic Parity Ratio</b> (DPR; min / max).
       Four-fifths rule: DPR &ge; 0.80 dan |DPD| &le; 0.10 dianggap acceptable.</p>
    {''.join(rows_html)}
    </body></html>"""
    path.write_text(html)


if __name__ == "__main__":
    main()
