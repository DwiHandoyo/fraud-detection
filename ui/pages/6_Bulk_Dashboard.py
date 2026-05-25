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
# Model accuracy vs user_label (ground truth from isFraud column or Bulk Audit)
# Only shown when at least one row has user_label set.
# ============================================================
def _label_str(predicted_label):
    if predicted_label == 1:
        return "fraud"
    if predicted_label == 0:
        return "legit"
    return None


labeled_rows = read_bulk_results(job_id, limit=10_000) if job_id is not None else []
if not labeled_rows and job_id is None:
    # Aggregate mode: pull recent N from any job
    for j in jobs[:20]:
        labeled_rows.extend(read_bulk_results(j["job_id"], limit=10_000))

eval_df = pd.DataFrame([
    r for r in labeled_rows
    if r.get("status") == "completed"
    and r.get("user_label") in ("fraud", "legit")
    and r.get("predicted_label") in (0, 1)
])

if not eval_df.empty:
    st.divider()
    st.subheader("Model accuracy vs ground truth")
    eval_df["pred_str"] = eval_df["predicted_label"].apply(_label_str)
    tp = int(((eval_df["pred_str"] == "fraud") & (eval_df["user_label"] == "fraud")).sum())
    fp = int(((eval_df["pred_str"] == "fraud") & (eval_df["user_label"] == "legit")).sum())
    fn = int(((eval_df["pred_str"] == "legit") & (eval_df["user_label"] == "fraud")).sum())
    tn = int(((eval_df["pred_str"] == "legit") & (eval_df["user_label"] == "legit")).sum())
    n = tp + fp + fn + tn

    accuracy = (tp + tn) / n if n else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Evaluated", n)
    m2.metric("Accuracy", f"{accuracy:.1%}")
    m3.metric("Precision", f"{precision:.1%}")
    m4.metric("Recall", f"{recall:.1%}")
    m5.metric("F1", f"{f1:.1%}")

    cm_left, cm_right = st.columns(2)
    with cm_left:
        st.caption("Confusion matrix (rows = actual, cols = predicted)")
        cm_df = pd.DataFrame(
            [[tn, fp], [fn, tp]],
            index=pd.Index(["actual legit", "actual fraud"], name=""),
            columns=["pred legit", "pred fraud"],
        )
        st.dataframe(cm_df, use_container_width=True)
    with cm_right:
        st.caption("Heatmap")
        heat_df = pd.DataFrame([
            {"actual": "legit", "predicted": "legit", "n": tn},
            {"actual": "legit", "predicted": "fraud", "n": fp},
            {"actual": "fraud", "predicted": "legit", "n": fn},
            {"actual": "fraud", "predicted": "fraud", "n": tp},
        ])
        chart_cm = (
            alt.Chart(heat_df)
            .mark_rect()
            .encode(
                x=alt.X("predicted:N", sort=["legit", "fraud"]),
                y=alt.Y("actual:N", sort=["legit", "fraud"]),
                color=alt.Color("n:Q", scale=alt.Scale(scheme="blues")),
                tooltip=["actual", "predicted", "n"],
            )
            .properties(height=180)
        )
        text_cm = chart_cm.mark_text(baseline="middle", fontSize=18).encode(
            text="n:Q",
            color=alt.condition(
                "datum.n > " + str(max(tn, fp, fn, tp) / 2),
                alt.value("white"), alt.value("black"),
            ),
        )
        st.altair_chart(chart_cm + text_cm, use_container_width=True)


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
