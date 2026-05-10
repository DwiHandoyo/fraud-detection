"""Human review UI for fraud predictions.

Single-file Streamlit app that calls ml-service /predict_by_id and
/explain_by_id, renders the prediction with a per-feature contribution chart,
and lets a human reviewer accept / reject / flag the decision.

Reviewer decisions are appended to ./decisions.jsonl for audit trail.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

ML_SERVICE_URL = os.getenv("ML_SERVICE_URL", "http://localhost:8000")
DECISIONS_LOG = Path(os.getenv("DECISIONS_LOG", "decisions.jsonl"))
LOW_CONF_RANGE = (0.30, 0.70)
TIMEOUT = 10


# ---------- API client ----------


def predict(transaction_id: int) -> dict:
    r = requests.post(
        f"{ML_SERVICE_URL}/predict_by_id",
        json={"transaction_id": transaction_id},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return r.json()


def explain(transaction_id: int) -> dict:
    r = requests.post(
        f"{ML_SERVICE_URL}/explain_by_id",
        json={"transaction_id": transaction_id},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return r.json()


def health() -> dict:
    r = requests.get(f"{ML_SERVICE_URL}/health", timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


# ---------- Audit log ----------


def log_decision(transaction_id: int, decision: str, prediction: dict, note: str = "") -> None:
    record = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "transaction_id": transaction_id,
        "decision": decision,
        "model_proba": prediction.get("fraud_proba"),
        "model_label": prediction.get("predicted_label"),
        "model": prediction.get("model"),
        "note": note,
    }
    with DECISIONS_LOG.open("a") as f:
        f.write(json.dumps(record) + "\n")


# ---------- UI ----------


st.set_page_config(page_title="Fraud Review", page_icon=":mag:", layout="wide")
st.title("Fraud Detection — Human Review")
st.caption(
    f"Calls ml-service at `{ML_SERVICE_URL}`. "
    "Sample TransactionIDs to try: 2987004, 2987008, 2987010, 2987011, 2987016."
)

# Sidebar — service health.
with st.sidebar:
    st.header("Service status")
    try:
        h = health()
        st.success("ml-service: ok")
        st.code(json.dumps(h, indent=2), language="json")
    except Exception as e:
        st.error(f"ml-service unreachable: {e}")
        st.stop()

    st.header("Audit log")
    if DECISIONS_LOG.exists():
        st.caption(f"{sum(1 for _ in DECISIONS_LOG.open())} decisions recorded")
    else:
        st.caption("no decisions yet")

# Input row.
col_in, col_btn = st.columns([3, 1])
with col_in:
    txn_id_str = st.text_input("Transaction ID", value="2987004")
with col_btn:
    st.write("")  # spacer
    score_btn = st.button("Score", type="primary", use_container_width=True)

if score_btn:
    try:
        txn_id = int(txn_id_str)
    except ValueError:
        st.error("Transaction ID must be an integer.")
        st.stop()

    with st.spinner("Calling ml-service..."):
        try:
            pred = predict(txn_id)
            expl = explain(txn_id)
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                st.error(f"transaction_id {txn_id} not found in feature store.")
            else:
                st.error(f"ml-service error: {e}")
            st.stop()

    st.session_state["txn_id"] = txn_id
    st.session_state["pred"] = pred
    st.session_state["expl"] = expl

# ---------- Render results ----------

if "pred" in st.session_state:
    txn_id = st.session_state["txn_id"]
    pred = st.session_state["pred"]
    expl = st.session_state["expl"]

    st.divider()

    # Top metrics row.
    proba = pred["fraud_proba"]
    low_conf = LOW_CONF_RANGE[0] <= proba <= LOW_CONF_RANGE[1]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Fraud probability", f"{proba:.3f}")
    c2.metric("Predicted label", "FRAUD" if pred["predicted_label"] == 1 else "LEGIT")
    c3.metric("Threshold", f"{pred['threshold']:.2f}")
    c4.metric("Model", f"{pred['model']}/{pred['version']}")

    if low_conf:
        st.warning(
            "**LOW CONFIDENCE** — probability is in the gray zone "
            f"({LOW_CONF_RANGE[0]} – {LOW_CONF_RANGE[1]}). Review carefully."
        )

    # Explanation.
    st.subheader("Per-feature contribution (EBM)")
    contribs = expl["contributions"]
    df = pd.DataFrame(contribs)
    df["abs"] = df["contribution"].abs()
    df = df.sort_values("abs", ascending=True).tail(15)  # top 15 by magnitude

    df["label"] = df.apply(lambda r: f"{r['feature']} = {r['value']}", axis=1)

    # Render as horizontal bar chart with red/green.
    import altair as alt

    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("contribution:Q", title="Contribution to log-odds(fraud)"),
            y=alt.Y("label:N", sort="-x", title=None),
            color=alt.condition(
                alt.datum.contribution > 0,
                alt.value("#dc2626"),  # red — fraud-pushing
                alt.value("#16a34a"),  # green — protective
            ),
            tooltip=["feature", "value", "contribution"],
        )
        .properties(height=400)
    )
    st.altair_chart(chart, use_container_width=True)
    st.caption("Red = pushes prediction toward fraud. Green = pushes away from fraud.")

    # Reviewer decision panel.
    st.divider()
    st.subheader("Reviewer decision")
    note = st.text_input("Note (optional)")
    d1, d2, d3, _ = st.columns([1, 1, 1, 3])
    if d1.button("Approve as legit", use_container_width=True):
        log_decision(txn_id, "approve_legit", pred, note)
        st.success("Logged: approve_legit")
    if d2.button("Confirm fraud", use_container_width=True):
        log_decision(txn_id, "confirm_fraud", pred, note)
        st.success("Logged: confirm_fraud")
    if d3.button("Need more info", use_container_width=True):
        log_decision(txn_id, "need_more_info", pred, note)
        st.info("Logged: need_more_info")
