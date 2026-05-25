"""Page 9 — Decision Detail.

Detail per-row dari Audit Log. Klik baris di Audit Log, lalu "View detail" ke
sini. Tampilkan decision row + related bulk_predictions + predictions rows
untuk transaction_id yang sama.
"""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Decision Detail", layout="wide")
st.title("Decision Detail")

try:
    from db import AVAILABLE as DB_AVAILABLE, get_decision_detail
except Exception as e:
    DB_AVAILABLE = False
    st.error(f"Database not available: {e}")
    st.stop()

if not DB_AVAILABLE:
    st.error("PostgreSQL belum tersedia.")
    st.stop()

decision_id = st.session_state.get("detail_decision_id")
if decision_id is None:
    st.info("Belum ada baris yang dipilih. Buka **Audit Log**, klik baris, lalu klik 'View detail'.")
    if st.button("Back to Audit Log"):
        st.switch_page("pages/3_Audit_Log.py")
    st.stop()

detail = get_decision_detail(int(decision_id))
if detail is None:
    st.error(f"Decision id={decision_id} tidak ditemukan.")
    if st.button("Back to Audit Log"):
        st.switch_page("pages/3_Audit_Log.py")
    st.stop()

d = detail["decision"]
bp = detail["bulk_predictions"]
preds = detail["predictions"]

# Top-bar with key facts.
c1, c2, c3, c4 = st.columns(4)
c1.metric("Decision", d.get("decision") or "-")
c2.metric("Source", d.get("source") or "-")
proba = d.get("model_proba")
c3.metric("Model proba", f"{proba:.3f}" if proba is not None else "-")
label = d.get("model_label")
c4.metric("Model label", str(label) if label is not None else "-")

st.caption(
    f"decision_id={d['id']}  |  transaction_id={d.get('transaction_id') or '-'}  |  "
    f"ts={d.get('ts')}  |  model={d.get('model') or '-'}"
)
if d.get("note"):
    st.info(f"Reviewer note: {d['note']}")

st.divider()

# Decision row full.
st.subheader("Decision row")
st.dataframe(pd.DataFrame([d]), use_container_width=True, hide_index=True)

# Bulk prediction (kalau source=bulk).
if bp is not None:
    st.divider()
    st.subheader("Linked bulk_predictions row")
    bp_view = {k: v for k, v in bp.items() if k != "request_payload"}
    st.dataframe(pd.DataFrame([bp_view]), use_container_width=True, hide_index=True)

    rp = bp.get("request_payload")
    if rp:
        with st.expander("Feature payload (raw)", expanded=False):
            if isinstance(rp, str):
                try:
                    rp = json.loads(rp)
                except Exception:
                    pass
            st.json(rp)

# Predictions history untuk transaction_id ini.
if preds:
    st.divider()
    st.subheader(f"Prediction history (last {len(preds)})")
    pred_df = pd.DataFrame(preds)
    cols = [c for c in ["ts", "source", "model", "model_version", "fraud_proba", "predicted_label"] if c in pred_df.columns]
    st.dataframe(
        pred_df[cols].style.format({"fraud_proba": "{:.3f}"}, na_rep="-"),
        use_container_width=True,
        hide_index=True,
    )
    if "request_payload" in pred_df.columns:
        with st.expander("Most recent prediction payload (raw)", expanded=False):
            rp = pred_df.iloc[0]["request_payload"]
            if isinstance(rp, str):
                try:
                    rp = json.loads(rp)
                except Exception:
                    pass
            st.json(rp)
elif d.get("transaction_id"):
    st.divider()
    st.caption(f"Tidak ada entry di `predictions` table untuk transaction_id={d['transaction_id']}.")

st.divider()
if st.button("Back to Audit Log"):
    st.switch_page("pages/3_Audit_Log.py")
