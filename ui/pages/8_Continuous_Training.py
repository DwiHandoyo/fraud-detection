"""Page 8 — Continuous Training.

Trigger retraining from human-corrected labels (bulk_predictions.user_label),
view training history, promote a challenger model to champion.

Spec compliance: Ops 4a (CI/CD + continuous training) + Arch 2c (human-in-loop).
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from api_client import hide_sidebar_pages

st.set_page_config(page_title="Continuous Training", layout="wide")
hide_sidebar_pages("Decision_Detail")
st.title("Continuous Training")

try:
    from db import (
        AVAILABLE as DB_AVAILABLE,
        labeled_row_counts,
        list_training_jobs,
        mark_training_job,
    )
except Exception as e:
    DB_AVAILABLE = False
    st.error(f"Database not available: {e}")
    st.stop()

if not DB_AVAILABLE:
    st.error("PostgreSQL belum tersedia. Halaman ini butuh DB.")
    st.stop()

st.caption(
    "Retrain LGBM dari koreksi reviewer di Bulk Audit. "
    "Pipeline: feedback → windowing → train → versioned pkl → manual promote."
)


def _promote_instructions(job: dict) -> None:
    """Show the manual steps needed to activate the promoted model."""
    st.info(
        f"**To activate the new model:**\n\n"
        f"1. Edit `ml-service/config.yaml`:\n\n"
        f"```yaml\n"
        f"predictor:\n"
        f"  path: {job['model_path']}\n"
        f"  version: \"{job['model_version']}\"\n"
        f"explainer:\n"
        f"  path: {job['model_path']}\n"
        f"  version: \"{job['model_version']}\"\n"
        f"```\n\n"
        f"2. Restart ml-service:\n\n"
        f"```bash\n"
        f"docker compose restart ml-service\n"
        f"```\n\n"
        f"Or symlink `lgbm_best.pkl` to point at this version:\n"
        f"```bash\n"
        f"cd ml-service/models && ln -sf $(basename {job['model_path']}) lgbm_best.pkl\n"
        f"```"
    )

# ============================================================
# Section 1: Feedback inventory
# ============================================================
st.subheader("Labeled data inventory")
counts = labeled_row_counts()
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total labeled", counts.get("total_labeled") or 0)
c2.metric("Fraud", counts.get("fraud") or 0)
c3.metric("Legit", counts.get("legit") or 0)
c4.metric("Review", counts.get("review") or 0)
c5.metric("Blocked", counts.get("blocked") or 0)

trainable = (counts.get("fraud") or 0) + (counts.get("legit") or 0)
if trainable == 0:
    st.warning(
        "Belum ada label `fraud` atau `legit`. Label rows di Bulk Audit dulu "
        "(Page 7) sebelum retrain."
    )

st.divider()

# ============================================================
# Section 2: Trigger retrain
# ============================================================
st.subheader("Trigger retrain")

col_l, col_r = st.columns(2)
with col_l:
    recent_days = st.slider(
        "Recent window (days)", min_value=7, max_value=180, value=30, step=7,
        help="Labels within this window = NEW (100% used)",
    )
    history_sample = st.slider(
        "History sample rate", min_value=0.05, max_value=1.0, value=0.20, step=0.05,
        help="Sample rate for labels older than window = OLD",
    )
with col_r:
    min_new_labels = st.number_input(
        "Min new labels required", min_value=1, max_value=10_000, value=5,
        help="Refuse retrain if fewer than N new labels.",
    )
    quick_mode = st.checkbox(
        "Quick mode (n_estimators=200, no early stop)",
        value=True,
        help="Smoke test mode. Uncheck for full training.",
    )

notes = st.text_input("Optional note (saved to training_jobs.notes)")

if trainable < min_new_labels:
    st.warning(
        f"Tombol Retrain di-disable: cuma **{trainable}** trainable rows (fraud + legit), "
        f"butuh minimal **{min_new_labels}**. "
        f"Pilihan: turunkan **Min new labels required** ke ≤{trainable}, atau label "
        f"lebih banyak rows di Bulk Audit."
    )

if st.button("Trigger Retrain", type="primary", use_container_width=True, disabled=trainable < min_new_labels):
    import threading
    import time as _time
    from api_client import trigger_retrain
    import requests as _req

    # Estimated duration so we can show fake progress that feels alive.
    # Quick mode ~30 s, full training ~3–5 min on the small demo dataset.
    expected_seconds = 30 if quick_mode else 240

    result_holder: dict = {}

    def _do_retrain():
        try:
            result_holder["result"] = trigger_retrain(
                recent_days=recent_days,
                history_sample_rate=history_sample,
                min_new_labels=min_new_labels,
                quick=quick_mode,
                notes=notes,
            )
        except _req.HTTPError as e:
            result_holder["error"] = (
                f"Retrain endpoint failed: {e}"
                + (f"\n{e.response.text[:500]}" if e.response is not None else "")
            )
        except _req.RequestException as e:
            result_holder["error"] = f"Connection to ml-service failed: {e}"
        except Exception as e:  # noqa: BLE001
            result_holder["error"] = f"Unexpected error: {e}"

    thread = threading.Thread(target=_do_retrain, daemon=True)
    thread.start()

    progress = st.progress(0.0)
    status = st.empty()

    elapsed = 0.0
    while thread.is_alive():
        pct = min(elapsed / expected_seconds, 0.95)  # cap at 95% until real result
        progress.progress(pct)
        status.markdown(f"**Retraining... {int(pct * 100)}%**  ·  elapsed {int(elapsed)}s")
        _time.sleep(1)
        elapsed += 1

    thread.join()
    progress.progress(1.0)
    status.empty()

    if "error" in result_holder:
        st.error(result_holder["error"])
    else:
        result = result_holder["result"]
        if result["status"] == "success":
            st.success(f"Retrain completed in {int(elapsed)}s.")
        else:
            st.error(f"Retrain failed (returncode={result['returncode']})")
        st.info("Scroll ke 'Training jobs history' di bawah — job baru sudah masuk.")

st.divider()

# ============================================================
# Section 3: Training jobs history
# ============================================================
st.subheader("Training jobs history")
jobs = list_training_jobs(limit=50)

if not jobs:
    st.info("Belum ada training job. Trigger di atas untuk mulai.")
else:
    df = pd.DataFrame(jobs)
    df["ts"] = pd.to_datetime(df["ts"])
    display_cols = ["id", "ts", "model_version", "n_new_labels", "n_old_labels",
                    "fraud_rate", "val_auc", "train_seconds", "status", "notes"]
    display_cols = [c for c in display_cols if c in df.columns]
    st.dataframe(
        df[display_cols].style.format({
            "fraud_rate": "{:.3f}",
            "val_auc": "{:.4f}",
            "train_seconds": "{:.1f}",
        }, na_rep="-"),
        use_container_width=True,
        height=300,
    )

# ============================================================
# Section 4: Promote a model
# ============================================================
st.divider()
st.subheader("Promote a trained model to Champion")

if not jobs:
    st.info("No trained models to promote yet.")
else:
    trained = [j for j in jobs if j.get("status") == "trained"]
    if not trained:
        st.info("Tidak ada model dengan status `trained` (sudah dipromote atau di-reject semua).")
    else:
        options = {
            f"#{j['id']} — {j['model_version']} — AUC {j.get('val_auc', 0):.4f} "
            f"({j['n_new_labels']} new + {j['n_old_labels']} old)": j
            for j in trained
        }
        sel = st.selectbox("Select trained model", list(options.keys()))
        chosen = options[sel]

        pc1, pc2, pc3 = st.columns([1, 1, 4])
        if pc1.button("Promote to Champion", type="primary"):
            ok = mark_training_job(chosen["id"], status="promoted", promoted_by="ui")
            if ok:
                _promote_instructions(chosen)
                st.success(f"Marked job #{chosen['id']} as promoted in DB.")
            else:
                st.error("Failed to update DB.")
        if pc2.button("Reject"):
            ok = mark_training_job(chosen["id"], status="rejected", promoted_by="ui")
            if ok:
                st.warning(f"Marked job #{chosen['id']} as rejected.")
                st.rerun()
