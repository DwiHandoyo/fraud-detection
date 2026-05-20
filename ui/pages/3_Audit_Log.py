"""Page 3 — Audit Log.

Tampilkan history keputusan reviewer. Sumber data: Postgres jika tersedia,
fallback ke decisions.jsonl. Memenuhi spec RAI 5b (audit trail).
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from api_client import DECISIONS_LOG, read_decisions

st.set_page_config(page_title="Audit Log", layout="wide")
st.title("Audit Log")

# Indicate data source.
try:
    from db import AVAILABLE as DB_AVAILABLE
except Exception:
    DB_AVAILABLE = False

if DB_AVAILABLE:
    st.caption("Source: **PostgreSQL** `decisions` table (filterable via SQL)")
else:
    st.caption(f"Source: **JSONL fallback** `{DECISIONS_LOG}` (DB unavailable)")

# Filters (sent to SQL if DB available).
fc1, fc2, fc3 = st.columns([2, 2, 3])
with fc1:
    decision_filter = st.multiselect(
        "Decision",
        options=["approve_legit", "confirm_fraud", "need_more_info"],
        default=[],
        placeholder="All decisions",
    )
with fc2:
    source_filter = st.multiselect(
        "Source",
        options=["by_id", "whatif"],
        default=[],
        placeholder="All sources",
    )
with fc3:
    search = st.text_input("Search by transaction_id or note", value="")

# Query.
records = read_decisions(
    limit=500,
    decision_filter=decision_filter or None,
    source_filter=source_filter or None,
    search=search,
)

if not records:
    st.info("Belum ada keputusan tercatat. Coba Score by ID atau What-if Scoring "
            "lalu klik Approve / Confirm / More-info.")
    st.stop()

df = pd.DataFrame(records)
df["ts"] = pd.to_datetime(df["ts"])

# Metrics.
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total (filtered)", len(df))
c2.metric("Approved legit", (df["decision"] == "approve_legit").sum())
c3.metric("Confirmed fraud", (df["decision"] == "confirm_fraud").sum())
c4.metric("Need more info", (df["decision"] == "need_more_info").sum())

st.divider()

# Table.
display_cols = ["ts", "transaction_id", "source", "decision", "model_proba", "model_label", "model", "note"]
display_cols = [c for c in display_cols if c in df.columns]
st.dataframe(
    df[display_cols].style.format({"model_proba": "{:.3f}"}, na_rep="-"),
    use_container_width=True,
    height=500,
)

st.divider()
csv = df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download filtered as CSV",
    data=csv,
    file_name="decisions_export.csv",
    mime="text/csv",
)
