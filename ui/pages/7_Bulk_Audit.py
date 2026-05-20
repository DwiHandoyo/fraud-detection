"""Page 7 — Bulk Audit.

Reviewer pilih bulk job, filter rows, beri label/block secara massal.
Label tersimpan di kolom `user_label` di tabel bulk_predictions.

Labels:
  - fraud    : confirmed fraud
  - legit    : confirmed legit
  - review   : need investigation
  - blocked  : block this entity (e.g. suspicious card/email pattern)
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Bulk Audit", layout="wide")
st.title("Bulk Audit")

try:
    from db import (
        AVAILABLE as DB_AVAILABLE,
        label_bulk_rows,
        list_bulk_jobs,
        read_bulk_results,
    )
except Exception as e:
    DB_AVAILABLE = False
    st.error(f"Database not available: {e}")
    st.stop()

if not DB_AVAILABLE:
    st.error("PostgreSQL belum tersedia.")
    st.stop()

st.caption(
    "Pilih bulk job, filter rows, centang yang ingin diberi label, lalu "
    "pilih label di tombol bawah. Label disimpan di Postgres dan tampil "
    "di Dashboard."
)

# ============================================================
# Job selector
# ============================================================
jobs = list_bulk_jobs()
if not jobs:
    st.info("Belum ada bulk job. Upload CSV di Bulk Predict dulu.")
    st.stop()

job_options = {
    f"{j['job_id']} — {j['filename']} ({j['completed']} completed, {j['labeled']} labeled)": j["job_id"]
    for j in jobs
}
sel = st.selectbox("Job", list(job_options.keys()))
job_id = job_options[sel]

# ============================================================
# Filters
# ============================================================
fc1, fc2, fc3 = st.columns(3)
with fc1:
    status_filter = st.multiselect(
        "Status",
        options=["pending", "completed", "failed"],
        default=["completed"],
    )
with fc2:
    label_filter_raw = st.multiselect(
        "User label",
        options=["fraud", "legit", "review", "blocked", "<unlabeled>"],
        default=[],
        placeholder="All",
    )
with fc3:
    proba_range = st.slider(
        "fraud_proba range", 0.0, 1.0, (0.0, 1.0), step=0.05,
    )

# Convert "<unlabeled>" sentinel — we model it as NULL in DB.
include_unlabeled = "<unlabeled>" in label_filter_raw
explicit_labels = [l for l in label_filter_raw if l != "<unlabeled>"]

# Query DB — we don't push <unlabeled> filter to SQL, handle in pandas.
rows = read_bulk_results(
    job_id=job_id,
    status_filter=status_filter or None,
    label_filter=explicit_labels or None,
    proba_min=proba_range[0],
    proba_max=proba_range[1],
    limit=2000,
)

if not rows:
    st.warning("Tidak ada row yang match filter.")
    st.stop()

df = pd.DataFrame(rows)

# If user asked for <unlabeled> AND no explicit labels, show only NULL.
# If user asked for <unlabeled> AND explicit, show union (already in df from explicit + need to re-query for nulls).
if include_unlabeled and not explicit_labels:
    df = df[df["user_label"].isna()]
elif include_unlabeled and explicit_labels:
    # Re-query for nulls too and merge.
    nulls = pd.DataFrame(
        read_bulk_results(job_id=job_id, status_filter=status_filter or None,
                          proba_min=proba_range[0], proba_max=proba_range[1], limit=2000)
    )
    nulls = nulls[nulls["user_label"].isna()]
    df = pd.concat([df, nulls]).drop_duplicates(subset=["id"]).reset_index(drop=True)

st.caption(f"Showing {len(df)} rows matching filter.")

# ============================================================
# Editable table with selection checkboxes
# ============================================================
df_display = df[["id", "row_idx", "transaction_id", "fraud_proba",
                 "predicted_label", "user_label", "user_note"]].copy()
df_display.insert(0, "Select", False)

edited = st.data_editor(
    df_display,
    use_container_width=True,
    height=450,
    column_config={
        "Select": st.column_config.CheckboxColumn("Select", default=False),
        "id": st.column_config.NumberColumn("id", disabled=True),
        "row_idx": st.column_config.NumberColumn("row_idx", disabled=True),
        "transaction_id": st.column_config.NumberColumn("transaction_id", disabled=True),
        "fraud_proba": st.column_config.NumberColumn("fraud_proba", format="%.3f", disabled=True),
        "predicted_label": st.column_config.NumberColumn("pred", disabled=True),
        "user_label": st.column_config.TextColumn("current label", disabled=True),
        "user_note": st.column_config.TextColumn("note", disabled=True),
    },
    hide_index=True,
    key="bulk_audit_editor",
)

selected_ids = edited[edited["Select"]]["id"].tolist()

# ============================================================
# Bulk label actions
# ============================================================
st.divider()
st.subheader(f"Apply label to {len(selected_ids)} selected rows")

note_in = st.text_input("Bulk note (optional)", key="bulk_note")

b1, b2, b3, b4, b5 = st.columns(5)


def _apply(label: str, color: str = "default") -> None:
    if not selected_ids:
        st.warning("Pilih row dulu (centang Select).")
        return
    n = label_bulk_rows(selected_ids, label, note_in)
    st.success(f"Updated {n} rows with label={label}.")
    st.session_state.pop("bulk_audit_editor", None)
    st.rerun()


if b1.button("Mark as FRAUD", use_container_width=True):
    _apply("fraud")
if b2.button("Mark as LEGIT", use_container_width=True):
    _apply("legit")
if b3.button("Mark as REVIEW", use_container_width=True):
    _apply("review")
if b4.button("BLOCK", use_container_width=True):
    _apply("blocked")
if b5.button("Clear label", use_container_width=True):
    if selected_ids:
        # Set user_label to NULL via passing explicit empty string and handling in DB.
        # Simpler: just label as NULL via a separate helper would be cleaner, but
        # for demo we treat 'clear' as setting to None — implement inline.
        from db import _engine, AVAILABLE as _AV
        from sqlalchemy import text as _text
        if _AV and _engine is not None:
            with _engine.begin() as conn:
                conn.execute(
                    _text("UPDATE bulk_predictions SET user_label = NULL, user_note = NULL, labeled_at = NULL WHERE id = ANY(:ids)"),
                    {"ids": selected_ids},
                )
            st.success(f"Cleared labels for {len(selected_ids)} rows.")
            st.session_state.pop("bulk_audit_editor", None)
            st.rerun()

# ============================================================
# Export labeled rows
# ============================================================
st.divider()
st.subheader("Export")
labeled_only = df[df["user_label"].notna()].copy()
st.caption(f"{len(labeled_only)} rows currently labeled in this view")
if not labeled_only.empty:
    csv = labeled_only.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download labeled rows as CSV",
        data=csv,
        file_name=f"bulk_labels_job_{job_id}.csv",
        mime="text/csv",
    )
