"""Page 4 — Service Status.

Detail health, model versioning, feature store info, dan link ke OpenAPI docs.
"""
from __future__ import annotations

import json

import streamlit as st

from api_client import DECISIONS_LOG, ML_SERVICE_URL, health

st.set_page_config(page_title="Service Status", layout="wide")
st.title("Service Status")
st.caption(f"Backend: `{ML_SERVICE_URL}`")

# Health.
st.subheader("ml-service health")
try:
    h = health()
    c1, c2, c3 = st.columns(3)
    c1.metric("Status", h["status"])
    c2.metric("Predictor", h["predictor"])
    c3.metric("Explainer", h["explainer"])

    c4, c5 = st.columns(2)
    c4.metric("Predictor features", h["n_predictor_features"])
    c5.metric("Explainer features", h["n_explainer_features"])

    with st.expander("Raw /health response", expanded=False):
        st.code(json.dumps(h, indent=2), language="json")
except Exception as e:
    st.error(f"ml-service unreachable: {e}")

# Audit log info.
st.divider()
st.subheader("Audit log")
if DECISIONS_LOG.exists():
    count = sum(1 for _ in DECISIONS_LOG.open())
    size_kb = DECISIONS_LOG.stat().st_size / 1024
    c1, c2, c3 = st.columns(3)
    c1.metric("File", DECISIONS_LOG.name)
    c2.metric("Records", count)
    c3.metric("Size (KB)", f"{size_kb:.1f}")
else:
    st.info(f"No decisions log yet at `{DECISIONS_LOG}`")

# External links.
st.divider()
st.subheader("Endpoints")
st.markdown(
    f"""
- [OpenAPI docs (interactive)]({ML_SERVICE_URL}/docs)
- [OpenAPI JSON]({ML_SERVICE_URL}/openapi.json)
- Health endpoint: `GET {ML_SERVICE_URL}/health`
- Predict (manual input): `POST {ML_SERVICE_URL}/predict`
- Predict (by ID): `POST {ML_SERVICE_URL}/predict_by_id`
- Explain (manual input): `POST {ML_SERVICE_URL}/explain`
- Explain (by ID): `POST {ML_SERVICE_URL}/explain_by_id`
"""
)

# Quick-test snippets.
st.divider()
st.subheader("Quick test (curl)")
st.code(f"""curl -X POST {ML_SERVICE_URL}/predict_by_id \\
  -H "Content-Type: application/json" \\
  -d '{{"transaction_id": 2987004}}'""", language="bash")

st.code(f"""curl -X POST {ML_SERVICE_URL}/predict \\
  -H "Content-Type: application/json" \\
  -d '{{"TransactionAmt": 250, "DeviceType": "mobile", "id_30": "iOS 11.1.2"}}'""", language="bash")
