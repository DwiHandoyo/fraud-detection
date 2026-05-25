"""Page 5 — Bulk Predict.

Upload CSV, simpan rows sebagai pending di DB, lalu proses satu per satu via
ml-service /predict. Status setiap row terlihat live (pending → completed |
failed). Resume support: kalau user navigasi keluar, pending rows tetap di DB
dan bisa di-resume.
"""
from __future__ import annotations

import io
import time

import pandas as pd
import requests
import streamlit as st

from api_client import ML_SERVICE_URL

st.set_page_config(page_title="Bulk Predict", layout="wide")
st.title("Bulk Predict")

try:
    from db import (
        AVAILABLE as DB_AVAILABLE,
        create_bulk_job,
        get_pending_rows,
        list_bulk_jobs,
        read_bulk_results,
        update_bulk_result,
    )
except Exception as e:
    DB_AVAILABLE = False
    st.error(f"Database not available: {e}")
    st.stop()

if not DB_AVAILABLE:
    st.error("PostgreSQL belum tersedia. Bulk Predict butuh DB untuk menyimpan "
             "pending rows. Start Postgres lalu refresh halaman ini.")
    st.stop()

st.caption(
    "Upload CSV dengan kolom fitur. Tiap row akan di-score via "
    f"`{ML_SERVICE_URL}/predict`. Status disimpan di Postgres — resumable "
    "kalau navigasi keluar."
)

# ============================================================
# Tab 1: Resume pending dari job sebelumnya
# Tab 2: Upload CSV baru
# ============================================================
tab_resume, tab_new = st.tabs(["Resume pending jobs", "Upload new CSV"])

with tab_resume:
    st.subheader("Pending jobs (resumable)")
    jobs = list_bulk_jobs()
    jobs_with_pending = [j for j in jobs if (j["pending"] or 0) > 0]
    if not jobs_with_pending:
        st.info("Tidak ada pending job. Upload CSV baru di tab kanan.")
    else:
        df_jobs = pd.DataFrame(jobs_with_pending)
        df_jobs["started_at"] = pd.to_datetime(df_jobs["started_at"])
        st.dataframe(
            df_jobs[["job_id", "filename", "started_at", "total", "pending", "completed", "failed"]],
            use_container_width=True,
        )
        resume_options = {
            f"{j['job_id']} — {j['filename']} ({j['pending']}/{j['total']} pending)": j["job_id"]
            for j in jobs_with_pending
        }
        sel = st.selectbox("Pilih job untuk di-resume", list(resume_options.keys()))
        if st.button("Resume processing", key="resume_btn"):
            st.session_state["active_job_id"] = resume_options[sel]
            st.session_state["resume_mode"] = True
            st.rerun()

with tab_new:
    st.subheader("Upload CSV baru")
    uploaded = st.file_uploader(
        "Choose CSV file",
        type=["csv"],
        help="Header = nama fitur. Kolom `transaction_id` opsional. "
             "Fitur yang dikenal: TransactionAmt, id_30, id_31, DeviceType, dst.",
    )

    if uploaded is not None:
        df_input = pd.read_csv(uploaded)
        st.success(f"Loaded `{uploaded.name}`: {len(df_input)} rows × {len(df_input.columns)} cols")

        with st.expander("Preview (first 5 rows)", expanded=True):
            st.dataframe(df_input.head(), use_container_width=True)

        st.write(f"Detected columns: `{', '.join(df_input.columns[:10])}`"
                 + (" ..." if len(df_input.columns) > 10 else ""))

        if "isFraud" in df_input.columns:
            n_labeled = df_input["isFraud"].notna().sum()
            st.info(f"`isFraud` column detected ({n_labeled} non-null). Akan dipakai sebagai "
                    "ground truth → otomatis muncul confusion matrix di Bulk Dashboard. "
                    "Kolom ini di-strip dari payload model.")

        if st.button("Submit for processing", type="primary"):
            rows = df_input.to_dict(orient="records")
            job_id = create_bulk_job(filename=uploaded.name, rows=rows)
            if job_id is None:
                st.error("Failed to create job.")
                st.stop()
            st.success(f"Created job {job_id} with {len(rows)} pending rows.")
            st.session_state["active_job_id"] = job_id
            st.session_state["resume_mode"] = False
            st.rerun()


# ============================================================
# Processing area — runs when active_job_id is set
# ============================================================
if "active_job_id" in st.session_state:
    job_id = st.session_state["active_job_id"]
    st.divider()
    st.subheader(f"Processing job {job_id}")

    pending = get_pending_rows(job_id, limit=10_000)

    if not pending:
        st.success("All rows processed.")
        results = read_bulk_results(job_id)
        if results:
            df_r = pd.DataFrame(results)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total", len(df_r))
            c2.metric("Completed", (df_r["status"] == "completed").sum())
            c3.metric("Failed", (df_r["status"] == "failed").sum())
            avg = df_r["fraud_proba"].mean()
            c4.metric("Avg fraud_proba", f"{avg:.3f}" if pd.notna(avg) else "—")

            st.dataframe(
                df_r[["row_idx", "transaction_id", "status", "fraud_proba",
                      "predicted_label", "model"]].style.format(
                          {"fraud_proba": "{:.3f}"}, na_rep="-"),
                use_container_width=True, height=400,
            )
        if st.button("Clear (back to upload)"):
            del st.session_state["active_job_id"]
            st.rerun()
        st.stop()

    st.write(f"**{len(pending)} pending rows** — processing now...")
    progress = st.progress(0.0)
    status_box = st.empty()

    completed = 0
    failed = 0
    total = len(pending)

    for i, row in enumerate(pending):
        payload = row["request_payload"] if isinstance(row["request_payload"], dict) else {}
        # Strip non-feature fields (we re-fetch by_id only if explicitly desired).
        payload = {k: v for k, v in payload.items() if v is not None and k != "transaction_id"}
        try:
            r = requests.post(
                f"{ML_SERVICE_URL}/predict", json=payload, timeout=10,
            )
            r.raise_for_status()
            resp = r.json()
            update_bulk_result(
                row["id"],
                fraud_proba=resp["fraud_proba"],
                predicted_label=resp["predicted_label"],
                model=resp["model"],
            )
            completed += 1
        except Exception as e:
            update_bulk_result(
                row["id"], fraud_proba=None, predicted_label=None,
                model=None, error=str(e)[:200],
            )
            failed += 1

        progress.progress((i + 1) / total)
        status_box.write(f"Processed {i + 1} / {total}  |  ✓ {completed}  |  ✗ {failed}")
        time.sleep(0.02)  # gentle pacing, prevents UI lockup

    progress.empty()
    status_box.success(f"Done. {completed} completed, {failed} failed.")
    time.sleep(1)
    st.rerun()
