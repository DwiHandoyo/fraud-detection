"""Adversarial robustness test — input perturbation against /predict.

Memenuhi spec IF5251 nomor QA 3b. Mengukur seberapa rentan model terhadap
perubahan kecil pada input. Untuk tiap fitur numerik, perturb +/- 1%, 5%, 10%
dan ukur berapa persen prediksi yang flip dari legit ke fraud (atau sebaliknya).

Output: reports/adversarial_report.json — tabel per-feature × magnitude.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
OUT = HERE / "reports" / "adversarial_report.json"

# Sample baseline transactions covering different fraud-probability ranges.
SAMPLE_TXN_IDS = [
    2987004, 2987008, 2987010, 2987011, 2987016, 2987017, 2987038, 2987040,
    2987048, 2987049, 2987057, 2987066, 2987069, 2987070, 2987072, 2987074,
    2987084, 2987093, 2987099, 2987100, 2987101, 2987104, 2987105,
]

NUMERIC_FIELDS = ["TransactionAmt", "id_02", "id_05", "id_11", "id_19", "id_20"]
PERTURBATIONS = [-0.10, -0.05, -0.01, 0.01, 0.05, 0.10]
THRESHOLD = 0.5


def predict_proba(base_url: str, payload: dict) -> float:
    r = requests.post(f"{base_url}/predict", json=payload, timeout=10)
    r.raise_for_status()
    return r.json()["fraud_proba"]


def predict_proba_by_id(base_url: str, txn_id: int) -> tuple[float, dict]:
    """Get baseline + the underlying feature payload from explain_by_id."""
    r = requests.post(f"{base_url}/predict_by_id", json={"transaction_id": txn_id}, timeout=10)
    r.raise_for_status()
    proba = r.json()["fraud_proba"]
    # Pull features via explain_by_id (gives us {feature, value, contribution}).
    e = requests.post(f"{base_url}/explain_by_id", json={"transaction_id": txn_id}, timeout=10)
    e.raise_for_status()
    features = {
        c["feature"]: c["value"]
        for c in e.json()["contributions"]
        if "&" not in c["feature"]  # skip pairwise interaction terms
    }
    return proba, features


def run(base_url: str) -> dict:
    print(f"[adversarial] base_url={base_url}")
    print(f"[adversarial] sampling {len(SAMPLE_TXN_IDS)} transactions")

    baselines = []
    for txn_id in SAMPLE_TXN_IDS:
        try:
            proba, feats = predict_proba_by_id(base_url, txn_id)
            baselines.append({"txn_id": txn_id, "baseline_proba": proba, "features": feats})
        except requests.HTTPError as e:
            print(f"  skip {txn_id}: {e}")

    print(f"[adversarial] obtained {len(baselines)} baselines")

    # Aggregate: per (feature, perturbation_pct) -> flip_count / total
    table = {f: {p: {"flips": 0, "total": 0} for p in PERTURBATIONS} for f in NUMERIC_FIELDS}

    for b in baselines:
        baseline_label = int(b["baseline_proba"] >= THRESHOLD)
        for field in NUMERIC_FIELDS:
            value = b["features"].get(field)
            if value is None or not isinstance(value, (int, float)):
                continue
            for pct in PERTURBATIONS:
                payload = copy.deepcopy({k: v for k, v in b["features"].items() if v is not None})
                payload[field] = float(value) * (1 + pct)
                try:
                    p = predict_proba(base_url, payload)
                except requests.HTTPError:
                    continue
                new_label = int(p >= THRESHOLD)
                table[field][pct]["total"] += 1
                if new_label != baseline_label:
                    table[field][pct]["flips"] += 1

    # Compute flip rate.
    summary = {}
    for field, pcts in table.items():
        summary[field] = {}
        for pct, counts in pcts.items():
            total = counts["total"]
            flips = counts["flips"]
            rate = flips / total if total else None
            summary[field][f"{pct:+.0%}"] = {
                "flips": flips, "total": total,
                "flip_rate": round(rate, 3) if rate is not None else None,
            }

    return {
        "n_baselines": len(baselines),
        "threshold": THRESHOLD,
        "results": summary,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    args = ap.parse_args()

    result = run(args.base_url)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))

    print("\n=== flip rate per feature × perturbation ===")
    print(f"{'feature':14s} " + "  ".join(f"{p:+.0%}".rjust(6) for p in PERTURBATIONS))
    for field, pcts in result["results"].items():
        cells = []
        for p in PERTURBATIONS:
            entry = pcts[f"{p:+.0%}"]
            r = entry["flip_rate"]
            cells.append(f"{r:.2%}".rjust(6) if r is not None else "  -   ")
        print(f"{field:14s} " + "  ".join(cells))
    print(f"\n[done] wrote {OUT}")


if __name__ == "__main__":
    main()
