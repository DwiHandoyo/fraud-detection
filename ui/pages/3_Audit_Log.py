"""Page 3 — Audit Log.

Tampilkan history keputusan reviewer dari decisions.jsonl. Memenuhi spec
RAI 5b (audit trail) dan FMEA UI-1 (allow review of past decisions).
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from api_client import DECISIONS_LOG, read_decisions

st.set_page_config(page_title="Audit Log", layout="wide")
st.title("Audit Log")
st.caption(f"Source file: `{DECISIONS_LOG}`")

records = read_decisions(limit=500)

if not records:
    st.info("Belum ada keputusan ter-log. Score satu transaksi di 'Score by ID' "
            "atau 'What-if Scoring', lalu klik Approve/Confirm/More-info.")
    st.stop()

df = pd.DataFrame(records)
df["ts"] = pd.to_datetime(df["ts"])

# Top metrics.
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total decisions", len(df))
c2.metric("Approved legit", (df["decision"] == "approve_legit").sum())
c3.metric("Confirmed fraud", (df["decision"] == "confirm_fraud").sum())
c4.metric("Need more info", (df["decision"] == "need_more_info").sum())

# Filters.
st.divider()
fc1, fc2 = st.columns(2)
with fc1:
    decision_filter = st.multiselect(
        "Filter by decision",
        options=df["decision"].unique().tolist(),
        default=df["decision"].unique().tolist(),
    )
with fc2:
    source_filter = st.multiselect(
        "Filter by source",
        options=df.get("source", pd.Series(["by_id"])).unique().tolist(),
        default=df.get("source", pd.Series(["by_id"])).unique().tolist(),
    )

filtered = df[df["decision"].isin(decision_filter)]
if "source" in filtered.columns:
    filtered = filtered[filtered["source"].isin(source_filter)]

# Search.
search = st.text_input("Search by transaction_id or note", value="")
if search:
    mask = (
        filtered["transaction_id"].astype(str).str.contains(search, na=False)
        | filtered.get("note", pd.Series([], dtype=str)).astype(str).str.contains(search, case=False, na=False)
    )
    filtered = filtered[mask]

st.caption(f"Showing {len(filtered)} of {len(df)} records")

# Table.
display_cols = ["ts", "transaction_id", "source", "decision", "model_proba", "model_label", "model", "note"]
display_cols = [c for c in display_cols if c in filtered.columns]
st.dataframe(
    filtered[display_cols].style.format({"model_proba": "{:.3f}"}),
    use_container_width=True,
    height=500,
)

# Export.
st.divider()
csv = filtered.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download filtered as CSV",
    data=csv,
    file_name="decisions_export.csv",
    mime="text/csv",
)
