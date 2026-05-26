"""Page 3 — Audit Log.

Tampilkan history keputusan reviewer. Sumber data: Postgres jika tersedia,
fallback ke decisions.jsonl. Memenuhi spec RAI 5b (audit trail).
"""
from __future__ import annotations

import pandas as pd
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

from api_client import DECISIONS_LOG, hide_sidebar_pages, read_decisions

st.set_page_config(page_title="Audit Log", layout="wide")
hide_sidebar_pages("Decision_Detail")
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
        options=["approve_legit", "confirm_fraud", "need_more_info", "blocked"],
        default=[],
        placeholder="All decisions",
    )
with fc2:
    source_filter = st.multiselect(
        "Source",
        options=["by_id", "whatif", "bulk"],
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
    st.info("Belum ada keputusan tercatat. Coba What-if Scoring "
            "lalu klik Approve / Confirm / More-info.")
    st.stop()

df = pd.DataFrame(records)
df["ts"] = pd.to_datetime(df["ts"])

# Metrics.
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total (filtered)", len(df))
c2.metric("Approved legit", (df["decision"] == "approve_legit").sum())
c3.metric("Confirmed fraud", (df["decision"] == "confirm_fraud").sum())
c4.metric("Need more info", (df["decision"] == "need_more_info").sum())
c5.metric("Blocked", (df["decision"] == "blocked").sum())

st.divider()
st.caption("Klik baris mana saja untuk lihat detail.")

# Table — clicking any cell selects the row and navigates to detail.
display_cols = ["ts", "transaction_id", "source", "decision", "model_proba", "model_label", "model", "note"]
display_cols = [c for c in display_cols if c in df.columns]

grid_df = df[["id", *display_cols]].copy()
grid_df["ts"] = grid_df["ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
grid_df["model_proba"] = grid_df["model_proba"].map(
    lambda v: f"{v:.3f}" if pd.notna(v) else "-"
)

gb = GridOptionsBuilder.from_dataframe(grid_df)
gb.configure_selection(selection_mode="single", use_checkbox=False)
gb.configure_default_column(sortable=True, filter=False, resizable=True)
gb.configure_column("id", hide=True)
grid_response = AgGrid(
    grid_df,
    gridOptions=gb.build(),
    update_mode=GridUpdateMode.SELECTION_CHANGED,
    height=500,
    fit_columns_on_grid_load=True,
    allow_unsafe_jscode=True,
    key="audit_log_aggrid",
)

selected = grid_response.get("selected_rows")
if selected is not None and len(selected) > 0:
    sel_row = selected.iloc[0] if hasattr(selected, "iloc") else selected[0]
    st.session_state["detail_decision_id"] = int(sel_row["id"])
    st.switch_page("pages/9_Decision_Detail.py")

st.divider()
csv = df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download filtered as CSV",
    data=csv,
    file_name="decisions_export.csv",
    mime="text/csv",
)
