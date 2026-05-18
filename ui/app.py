"""Fraud Detection — Landing page.

Streamlit multipage app. This file is the entry point; the actual feature
pages are auto-loaded from pages/ and listed in the left sidebar.

Run:
    streamlit run app.py
"""
from __future__ import annotations

import streamlit as st

from api_client import ML_SERVICE_URL, health

st.set_page_config(
    page_title="Fraud Detection — Home",
    page_icon=None,
    layout="wide",
)

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
| **Score by ID** | Reviewer mode — lookup transaksi yang sudah ada via Feast online store. Input: `transaction_id`. Cocok untuk demo flow production. |
| **What-if Scoring** | Audit / debug mode — input fitur manual untuk eksplorasi skenario hipotetis. Cocok untuk reproducing test cases atau exploring model behavior. |
| **Audit Log** | Lihat history keputusan reviewer (`decisions.jsonl`). Filter, search, export. |
| **Service Status** | Detail health, model version, feature store status, prediction log info. |

---

### Bagaimana sistem ini bekerja

```
[Score by ID]                          [What-if Scoring]
        |                                       |
        v                                       v
POST /predict_by_id              POST /predict
        |                                       |
        v                                       v
   Feast online store           Pakai fitur dari request body
   (33 features dari            (caller mengisi langsung)
   transaction_id)
        |                                       |
        +-----------+---------------------------+
                    |
                    v
            LGBM predictor
            + EBM explainer
                    |
                    v
            Response: probability,
            label, per-feature breakdown
```

### Tips

- Reviewer fraud sungguhan biasanya pakai **Score by ID** — transaksi sudah ada di sistem.
- **What-if Scoring** untuk audit "kalau amount lebih besar, fraud-nya berubah?".
- Sample TransactionID untuk dicoba: `2987004`, `2987008`, `2987010`, `2987016`, `2987017`.
"""
)
