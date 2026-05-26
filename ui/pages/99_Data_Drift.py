"""Page — Data Drift (Evidently report viewer).

Tampilkan ringkasan + full HTML Evidently report yang di-generate offline via
`monitoring/generate_report.py`. Page baca file dari volume read-only
`/monitoring/reports/`.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from api_client import hide_sidebar_pages

st.set_page_config(page_title="Data Drift", layout="wide")
hide_sidebar_pages("Decision_Detail")
st.title("Data Drift")
st.caption(
    "Output Evidently drift report. Reference = 50% baris pertama "
    "`identity.parquet`, current = 50% terakhir."
)

REPORT_DIR = Path("/monitoring/reports")
REPORT_HTML = REPORT_DIR / "drift_report.html"
REPORT_JSON = REPORT_DIR / "drift_summary.json"

if not REPORT_JSON.exists():
    st.warning(
        f"`{REPORT_JSON}` tidak ditemukan. Jalankan `python monitoring/generate_report.py` "
        "dulu untuk men-generate report."
    )
    st.stop()

# ============================================================
# Parse drift_summary.json
# ============================================================
try:
    summary = json.loads(REPORT_JSON.read_text())
    metrics = summary.get("metrics", [])
except Exception as e:
    st.error(f"Gagal parse {REPORT_JSON}: {e}")
    st.stop()

drifted_count = 0
drift_share = 0.0
rows = []
for m in metrics:
    name = m.get("metric_name", "")
    val = m.get("value")
    if name.startswith("DriftedColumnsCount"):
        if isinstance(val, dict):
            drifted_count = int(val.get("count", 0))
            drift_share = float(val.get("share", 0.0))
    elif name.startswith("ValueDrift("):
        try:
            col = name.split("column=")[1].split(",")[0]
            threshold = float(name.split("threshold=")[1].rstrip(")"))
        except (IndexError, ValueError):
            continue
        score = val if isinstance(val, (int, float)) else (val or {}).get("value")
        if score is None:
            continue
        rows.append({
            "column": col,
            "drift_score": float(score),
            "threshold": threshold,
            "drifted": "yes" if float(score) > threshold else "no",
        })

n_monitored = len(rows)
mtime = datetime.fromtimestamp(REPORT_JSON.stat().st_mtime, tz=timezone.utc)

# ============================================================
# Header metric cards
# ============================================================
c1, c2, c3, c4 = st.columns(4)
c1.metric("Columns monitored", n_monitored)
c2.metric("Columns drifted", drifted_count)
c3.metric("Drift rate", f"{drift_share * 100:.1f}%" if drift_share else f"{(drifted_count / n_monitored * 100) if n_monitored else 0:.1f}%")
c4.metric("Last generated", mtime.strftime("%Y-%m-%d %H:%M UTC"))

st.divider()

# ============================================================
# Per-column table
# ============================================================
if rows:
    st.subheader("Per-column drift score")
    df = pd.DataFrame(rows).sort_values("drift_score", ascending=False).reset_index(drop=True)
    st.dataframe(
        df.style.format({"drift_score": "{:.4f}", "threshold": "{:.2f}"}),
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("Tidak ada per-column ValueDrift metrics di JSON.")

st.divider()

# ============================================================
# Full Evidently HTML — lazy load di dalam expander
# ============================================================
if REPORT_HTML.exists():
    size_mb = REPORT_HTML.stat().st_size / (1024 * 1024)
    with st.expander(f"Show full Evidently report ({size_mb:.1f} MB iframe)", expanded=False):
        try:
            html_str = REPORT_HTML.read_text(encoding="utf-8")
            components.html(html_str, height=1200, scrolling=True)
        except Exception as e:
            st.error(f"Gagal render HTML report: {e}")
else:
    st.caption(f"`{REPORT_HTML}` tidak ditemukan, hanya JSON summary yang tersedia.")
