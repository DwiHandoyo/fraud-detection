"""Fraud Detection — Landing page.

Streamlit multipage app. This file is the entry point; the actual feature
pages are auto-loaded from pages/ and listed in the left sidebar.

Run:
    streamlit run app.py
"""
from __future__ import annotations

import streamlit as st

from api_client import ML_SERVICE_URL, health, hide_sidebar_pages

st.set_page_config(
    page_title="Fraud Detection — Home",
    page_icon=None,
    layout="wide",
)
hide_sidebar_pages("Decision_Detail")

st.title("Fraud Detection — Operator Console")
st.caption(f"Backend: `{ML_SERVICE_URL}`")

# Service status banner (compact).
try:
    h = health()
    st.success(f"ml-service OK | predictor={h['predictor']} | explainer={h['explainer']}")
except Exception as e:
    st.error(f"ml-service unreachable: {e}")
    st.info("Service Status page below may give more detail once available.")

st.divider()

st.markdown(
    """
### Pages (lihat sidebar kiri)

| Halaman | Tujuan |
|---------|--------|
| **What-if Scoring** | Audit / debug mode — input fitur manual untuk eksplorasi skenario hipotetis. Cocok untuk reproducing test cases atau exploring model behavior. |
| **Audit Log** | Lihat history keputusan reviewer (`decisions.jsonl`). Filter, search, export. |
| **Service Status** | Detail health, model version, feature store status, prediction log info. |

---

### Bagaimana sistem ini bekerja

```
[What-if Scoring]
        |
        v
POST /predict
        |
        v
Pakai fitur dari request body
(caller mengisi langsung)
        |
        v
LGBM predictor (+ SHAP explainer)
        |
        v
Response: probability, label,
per-feature breakdown
```

### Tips

- **What-if Scoring** untuk audit "kalau amount lebih besar, fraud-nya berubah?".
- Bulk Predict + Bulk Audit untuk batch scoring + reviewer labeling.
"""
)
