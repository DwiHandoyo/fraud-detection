"""Page 1 — Score by ID.

Lookup fitur dari Feast online store berdasarkan transaction_id, lalu
prediksi + explanation. Tutorial use case untuk reviewer fraud.
"""
from __future__ import annotations

import json

import requests
import streamlit as st

from api_client import (
    explain_by_id,
    health,
    log_decision,
    predict_by_id,
    render_explanation_chart,
    render_prediction_header,
)

st.set_page_config(page_title="Score by ID", layout="wide")
st.title("Score by ID")
st.caption(
    "Reviewer mode — input `transaction_id` saja, fitur otomatis di-lookup "
    "dari Feast online store. Sample: 2987004, 2987008, 2987010, 2987016."
)

# Sidebar — service health snapshot.
with st.sidebar:
    st.header("Service status")
    try:
        h = health()
        st.success("ml-service: ok")
        st.code(json.dumps(h, indent=2), language="json")
    except Exception as e:
        st.error(f"ml-service unreachable: {e}")
        st.stop()

# Input row.
col_in, col_btn = st.columns([3, 1])
with col_in:
    txn_id_str = st.text_input("Transaction ID", value="2987004")
with col_btn:
    st.write("")
    score_btn = st.button("Score", type="primary", use_container_width=True)

if score_btn:
    try:
        txn_id = int(txn_id_str)
    except ValueError:
        st.error("Transaction ID must be an integer.")
        st.stop()

    with st.spinner("Calling ml-service..."):
        try:
            pred = predict_by_id(txn_id)
            expl = explain_by_id(txn_id)
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                st.error(f"transaction_id {txn_id} not found in feature store.")
            else:
                st.error(f"ml-service error: {e}")
            st.stop()

    st.session_state["score_by_id_txn"] = txn_id
    st.session_state["score_by_id_pred"] = pred
    st.session_state["score_by_id_expl"] = expl

# Render results from session state (persists across reruns).
if "score_by_id_pred" in st.session_state:
    txn_id = st.session_state["score_by_id_txn"]
    pred = st.session_state["score_by_id_pred"]
    expl = st.session_state["score_by_id_expl"]

    st.divider()
    render_prediction_header(pred)

    st.subheader("Per-feature contribution (EBM)")
    render_explanation_chart(expl["contributions"])

    # Reviewer decision.
    st.divider()
    st.subheader("Reviewer decision")
    note = st.text_input("Note (optional)", key="note_by_id")
    d1, d2, d3, _ = st.columns([1, 1, 1, 3])
    if d1.button("Approve as legit", use_container_width=True):
        log_decision(txn_id, "approve_legit", pred, note, source="by_id")
        st.success("Logged: approve_legit")
    if d2.button("Confirm fraud", use_container_width=True):
        log_decision(txn_id, "confirm_fraud", pred, note, source="by_id")
        st.success("Logged: confirm_fraud")
    if d3.button("Need more info", use_container_width=True):
        log_decision(txn_id, "need_more_info", pred, note, source="by_id")
        st.info("Logged: need_more_info")
