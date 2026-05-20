"""Page 6 — Bulk Dashboard.

Visualisasi hasil bulk_predictions: distribusi probabilitas, breakdown status,
agregat user label per job. Read-only.
"""
from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Bulk Dashboard", layout="wide")
st.title("Bulk Dashboard")

try:
    from db import AVAILABLE as DB_AVAILABLE, dashboard_metrics, list_bulk_jobs, read_bulk_results
except Exception as e:
    DB_AVAILABLE = False
    st.error(f"Database not available: {e}")
    st.stop()

if not DB_AVAILABLE:
    st.error("PostgreSQL belum tersedia.")
    st.stop()

# ============================================================
# Job selector
# ============================================================
jobs = list_bulk_jobs()
if not jobs:
    st.info("Belum ada bulk job. Upload CSV di Bulk Predict dulu.")
    st.stop()

job_options = {"All jobs (aggregate)": None}
for j in jobs:
    label = f"{j['job_id']} — {j['filename']} ({j['total']} rows, {j['pending']} pending)"
    job_options[label] = j["job_id"]

sel = st.selectbox("Job", list(job_options.keys()))
job_id = job_options[sel]

# ============================================================
# Summary metrics
# ============================================================
metrics = dashboard_metrics(job_id=job_id)
summary = metrics.get("summary", {})

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total rows", summary.get("total") or 0)
c2.metric("Pending", summary.get("pending") or 0)
c3.metric("Completed", summary.get("completed") or 0)
c4.metric("Failed", summary.get("failed") or 0)

c5, c6, c7 = st.columns(3)
c5.metric("Predicted fraud", summary.get("predicted_fraud") or 0)
c6.metric("Predicted legit", summary.get("predicted_legit") or 0)
avg = summary.get("avg_proba")
c7.metric("Avg fraud_proba", f"{avg:.3f}" if avg is not None else "—")

# ============================================================
# Status pie
# ============================================================
st.divider()
left, right = st.columns(2)

with left:
    st.subheader("Status breakdown")
    status_df = pd.DataFrame([
        {"status": "pending", "n": summary.get("pending") or 0},
        {"status": "completed", "n": summary.get("completed") or 0},
        {"status": "failed", "n": summary.get("failed") or 0},
    ])
    status_df = status_df[status_df["n"] > 0]
    if not status_df.empty:
        chart_status = (
            alt.Chart(status_df)
            .mark_arc(innerRadius=50)
            .encode(
                theta="n:Q",
                color=alt.Color(
                    "status:N",
                    scale=alt.Scale(
                        domain=["completed", "pending", "failed"],
                        range=["#16a34a", "#eab308", "#dc2626"],
                    ),
                ),
                tooltip=["status", "n"],
            )
        )
        st.altair_chart(chart_status, use_container_width=True)
    else:
        st.caption("No rows yet.")

with right:
    st.subheader("User label breakdown")
    by_label = metrics.get("by_label", [])
    label_df = pd.DataFrame(by_label) if by_label else pd.DataFrame(columns=["user_label", "n"])
    if not label_df.empty and label_df["user_label"].notna().any():
        label_df["user_label"] = label_df["user_label"].fillna("unlabeled")
        chart_label = (
            alt.Chart(label_df)
            .mark_arc(innerRadius=50)
            .encode(
                theta="n:Q",
                color=alt.Color(
                    "user_label:N",
                    scale=alt.Scale(
                        domain=["fraud", "legit", "review", "blocked", "unlabeled"],
                        range=["#dc2626", "#16a34a", "#eab308", "#000000", "#94a3b8"],
                    ),
                ),
                tooltip=["user_label", "n"],
            )
        )
        st.altair_chart(chart_label, use_container_width=True)
    else:
        st.caption("Belum ada label. Pakai Bulk Audit untuk men-tag.")

# ============================================================
# Fraud probability histogram
# ============================================================
st.divider()
st.subheader("Fraud probability distribution (completed rows only)")
buckets = metrics.get("proba_buckets", [])
if buckets:
    bucket_df = pd.DataFrame(buckets)
    bucket_df["bucket_label"] = bucket_df["bucket"].apply(
        lambda b: f"{(b - 1) * 0.1:.1f}–{b * 0.1:.1f}" if b else "<0"
    )
    chart_hist = (
        alt.Chart(bucket_df)
        .mark_bar(color="#2563eb")
        .encode(
            x=alt.X("bucket_label:N", title="fraud_proba bucket", sort=None),
            y=alt.Y("n:Q", title="Count"),
            tooltip=["bucket_label", "n"],
        )
        .properties(height=300)
    )
    st.altair_chart(chart_hist, use_container_width=True)
else:
    st.caption("Tidak ada completed rows. Process bulk job dulu di Bulk Predict.")

# ============================================================
# Recent rows preview
# ============================================================
if job_id is not None:
    st.divider()
    st.subheader("Sample rows (latest 20)")
    rows = read_bulk_results(job_id, limit=20)
    if rows:
        df = pd.DataFrame(rows)
        cols = [c for c in ["row_idx", "transaction_id", "status", "fraud_proba",
                            "predicted_label", "user_label", "user_note"]
                if c in df.columns]
        st.dataframe(
            df[cols].style.format({"fraud_proba": "{:.3f}"}, na_rep="-"),
            use_container_width=True,
        )
