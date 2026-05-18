"""Shared HTTP client for ml-service.

Imported by all pages. Single source of base URL + timeouts + error handling.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import requests

ML_SERVICE_URL = os.getenv("ML_SERVICE_URL", "http://localhost:8000")
DECISIONS_LOG = Path(os.getenv("DECISIONS_LOG", "decisions.jsonl"))
TIMEOUT = 10
LOW_CONF_RANGE = (0.30, 0.70)


def health() -> dict:
    r = requests.get(f"{ML_SERVICE_URL}/health", timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def predict_by_id(transaction_id: int) -> dict:
    r = requests.post(
        f"{ML_SERVICE_URL}/predict_by_id",
        json={"transaction_id": transaction_id},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return r.json()


def explain_by_id(transaction_id: int) -> dict:
    r = requests.post(
        f"{ML_SERVICE_URL}/explain_by_id",
        json={"transaction_id": transaction_id},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return r.json()


def predict_manual(payload: dict) -> dict:
    r = requests.post(f"{ML_SERVICE_URL}/predict", json=payload, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def explain_manual(payload: dict) -> dict:
    r = requests.post(f"{ML_SERVICE_URL}/explain", json=payload, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def log_decision(transaction_id, decision: str, prediction: dict, note: str = "", source: str = "by_id") -> None:
    record = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "transaction_id": transaction_id,
        "source": source,
        "decision": decision,
        "model_proba": prediction.get("fraud_proba"),
        "model_label": prediction.get("predicted_label"),
        "model": prediction.get("model"),
        "note": note,
    }
    DECISIONS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with DECISIONS_LOG.open("a") as f:
        f.write(json.dumps(record) + "\n")


def read_decisions(limit: int = 200) -> list[dict]:
    if not DECISIONS_LOG.exists():
        return []
    lines = DECISIONS_LOG.read_text().strip().split("\n")
    records = [json.loads(line) for line in lines if line.strip()]
    return list(reversed(records))[:limit]


def render_explanation_chart(contributions: list[dict], top_n: int = 15):
    """Shared bar chart renderer. Imported by both scoring pages."""
    import altair as alt
    import pandas as pd
    import streamlit as st

    df = pd.DataFrame(contributions)
    df["abs"] = df["contribution"].abs()
    df = df.sort_values("abs", ascending=True).tail(top_n)
    df["label"] = df.apply(lambda r: f"{r['feature']} = {r['value']}", axis=1)

    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("contribution:Q", title="Contribution to log-odds(fraud)"),
            y=alt.Y("label:N", sort="-x", title=None),
            color=alt.condition(
                alt.datum.contribution > 0,
                alt.value("#dc2626"),
                alt.value("#16a34a"),
            ),
            tooltip=["feature", "value", "contribution"],
        )
        .properties(height=400)
    )
    st.altair_chart(chart, use_container_width=True)
    st.caption("Red = pushes toward fraud. Green = pushes away from fraud.")


def render_prediction_header(pred: dict):
    """Shared metrics row + low-confidence warning."""
    import streamlit as st

    proba = pred["fraud_proba"]
    low_conf = LOW_CONF_RANGE[0] <= proba <= LOW_CONF_RANGE[1]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Fraud probability", f"{proba:.3f}")
    c2.metric("Predicted label", "FRAUD" if pred["predicted_label"] == 1 else "LEGIT")
    c3.metric("Threshold", f"{pred['threshold']:.2f}")
    c4.metric("Model", f"{pred['model']}/{pred['version']}")

    if low_conf:
        st.warning(
            f"**LOW CONFIDENCE** — probability is in the gray zone "
            f"({LOW_CONF_RANGE[0]} - {LOW_CONF_RANGE[1]}). Review carefully."
        )
