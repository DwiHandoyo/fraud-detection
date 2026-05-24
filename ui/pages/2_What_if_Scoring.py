"""Page 2 — What-if Scoring.

Input fitur manual untuk eksplorasi skenario hipotetis. Cocok untuk:
  - Audit "kalau amount naik 10x, fraud-nya gimana?"
  - Reproducing test cases dari laporan
  - Demonstrating model behavior dengan fitur konkret
"""
from __future__ import annotations

import requests
import streamlit as st

from api_client import (
    explain_manual,
    log_decision,
    render_explanation_chart,
    render_prediction_header,
)

st.set_page_config(page_title="What-if Scoring", layout="wide")
st.title("What-if Scoring")
st.caption(
    "Audit / debug mode — isi fitur manual, lihat prediksi. "
    "Model: lgbm_best (107 fitur)."
)

# Default values — sample yang realistic dari train_identity.csv.
DEFAULTS = {
    "TransactionAmt": 100.0,
    "card4": "visa",
    "card6": "credit",
    "P_emaildomain": "gmail.com",
    "id_30": "Windows 10",
    "id_31": "chrome 62.0",
    "id_14": -300.0,
    "DeviceType": "desktop",
    "id_33": "1920x1080",
}

CARD4_OPTIONS = ["visa", "mastercard", "american express", "discover"]
CARD6_OPTIONS = ["credit", "debit", "debit or credit", "charge card"]
DEVICETYPE_OPTIONS = ["desktop", "mobile"]
ID30_OPTIONS = [
    "Windows 10", "Windows 7", "Windows 8.1", "iOS 11.1.2", "iOS 11.2.1",
    "iOS 11.4.1", "Mac OS X 10_11_6", "Mac OS X 10_12_6", "Mac OS X 10_13_5",
    "Android 7.0", "Android 7.1.1", "Android 8.0.0", "Linux", "other",
]
ID31_OPTIONS = [
    "chrome 62.0", "chrome 63.0", "chrome 65.0",
    "mobile safari 11.0", "safari generic", "ie 11.0 for desktop",
    "firefox 57.0", "edge 16.0", "samsung browser 6.2", "other",
]
ID33_OPTIONS = ["1920x1080", "1366x768", "1334x750", "1440x900", "2048x1536", "other"]

st.subheader("Transaction features")
with st.form("manual_scoring"):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Transaction**")
        TransactionAmt = st.number_input(
            "TransactionAmt (USD)", min_value=0.0, max_value=100000.0,
            value=DEFAULTS["TransactionAmt"], step=10.0,
        )
        card4 = st.selectbox("card4 (brand)", CARD4_OPTIONS, index=0)
        card6 = st.selectbox("card6 (type)", CARD6_OPTIONS, index=0)
        P_emaildomain = st.text_input("P_emaildomain", value=DEFAULTS["P_emaildomain"])

    with c2:
        st.markdown("**Device**")
        DeviceType = st.selectbox("DeviceType", DEVICETYPE_OPTIONS, index=0)
        id_30 = st.selectbox("id_30 (OS)", ID30_OPTIONS, index=0)
        id_31 = st.selectbox("id_31 (browser)", ID31_OPTIONS, index=0)
        id_33 = st.selectbox("id_33 (screen)", ID33_OPTIONS, index=0)

    with c3:
        st.markdown("**Context**")
        id_14 = st.number_input(
            "id_14 (timezone offset, minutes)",
            min_value=-720, max_value=840, value=int(DEFAULTS["id_14"]),
            step=60,
            help="-300 = US Eastern, -480 = US Pacific, 420 = Indonesia (WIB)",
        )
        id_11 = st.slider(
            "id_11 (rating score %)", min_value=0.0, max_value=100.0, value=100.0,
        )

    submitted = st.form_submit_button("Score", type="primary", use_container_width=True)

if submitted:
    payload = {
        "TransactionAmt": TransactionAmt,
        "card4": card4,
        "card6": card6,
        "P_emaildomain": P_emaildomain,
        "id_30": id_30,
        "id_31": id_31,
        "id_33": id_33,
        "DeviceType": DeviceType,
        "id_14": id_14,
        "id_11": id_11,
    }

    with st.spinner("Calling ml-service..."):
        try:
            expl = explain_manual(payload)
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 422:
                st.error(f"Validation failed: {e.response.json().get('detail')}")
            else:
                st.error(f"ml-service error: {e}")
            st.stop()

    # /explain returns fraud_proba directly from EBM — derive PredictResponse shape.
    pred = {
        "fraud_proba": expl["fraud_proba"],
        "predicted_label": int(expl["fraud_proba"] >= 0.5),
        "threshold": 0.5,
        "model": expl["model"],
        "version": expl["version"],
    }

    st.session_state["whatif_payload"] = payload
    st.session_state["whatif_pred"] = pred
    st.session_state["whatif_expl"] = expl

if "whatif_pred" in st.session_state:
    pred = st.session_state["whatif_pred"]
    expl = st.session_state["whatif_expl"]
    payload = st.session_state["whatif_payload"]

    st.divider()
    render_prediction_header(pred)

    with st.expander("Request payload (debugging)", expanded=False):
        st.json(payload)

    st.subheader("Per-feature contribution (EBM)")
    render_explanation_chart(expl["contributions"])

    st.divider()
    st.subheader("Audit decision")
    note = st.text_input("Note (optional)", key="note_whatif")
    d1, d2, d3, _ = st.columns([1, 1, 1, 3])
    if d1.button("Mark as legit", use_container_width=True):
        log_decision(None, "approve_legit", pred, note, source="whatif")
        st.success("Logged: approve_legit (what-if)")
    if d2.button("Mark as fraud", use_container_width=True):
        log_decision(None, "confirm_fraud", pred, note, source="whatif")
        st.success("Logged: confirm_fraud (what-if)")
    if d3.button("Need more info", use_container_width=True):
        log_decision(None, "need_more_info", pred, note, source="whatif")
        st.info("Logged: need_more_info (what-if)")
