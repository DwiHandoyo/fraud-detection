"""PostgreSQL bridge for the UI — insert decisions, read decisions.

Graceful fallback: if DATABASE_URL is unset or the DB is unreachable, helpers
return None / empty list. UI code uses unified helpers in api_client and
should not need to import this module directly.
"""
from __future__ import annotations

import logging
import os
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

logger = logging.getLogger("ui.db")

DATABASE_URL = os.getenv("DATABASE_URL")
_engine: Engine | None = None


def _init_engine() -> Engine | None:
    if not DATABASE_URL:
        return None
    try:
        eng = create_engine(DATABASE_URL, pool_pre_ping=True, pool_size=2, max_overflow=2)
        with eng.connect() as conn:
            conn.execute(text("SELECT 1"))
        return eng
    except Exception as e:
        logger.warning("UI DB connection failed: %s", e)
        return None


_engine = _init_engine()
AVAILABLE = _engine is not None


def insert_decision(
    *,
    transaction_id: int | None,
    source: str,
    decision: str,
    model_proba: float | None,
    model_label: int | None,
    model: str | None,
    note: str = "",
) -> int | None:
    if not AVAILABLE or _engine is None:
        return None
    try:
        with _engine.begin() as conn:
            result = conn.execute(
                text("""
                INSERT INTO decisions
                    (transaction_id, source, decision, model_proba, model_label, model, note)
                VALUES
                    (:transaction_id, :source, :decision, :model_proba, :model_label, :model, :note)
                RETURNING id
                """),
                {
                    "transaction_id": transaction_id,
                    "source": source,
                    "decision": decision,
                    "model_proba": model_proba,
                    "model_label": model_label,
                    "model": model,
                    "note": note,
                },
            )
            return int(result.scalar())
    except Exception as e:
        logger.warning("insert_decision failed: %s", e)
        return None


def read_decisions(limit: int = 500, decision_filter: list[str] | None = None,
                   source_filter: list[str] | None = None, search: str = "") -> list[dict[str, Any]]:
    if not AVAILABLE or _engine is None:
        return []
    try:
        sql = "SELECT id, ts, transaction_id, source, decision, model_proba, model_label, model, note FROM decisions"
        clauses, params = [], {}
        if decision_filter:
            clauses.append("decision = ANY(:decision_filter)")
            params["decision_filter"] = decision_filter
        if source_filter:
            clauses.append("source = ANY(:source_filter)")
            params["source_filter"] = source_filter
        if search:
            clauses.append("(CAST(transaction_id AS TEXT) ILIKE :q OR COALESCE(note, '') ILIKE :q)")
            params["q"] = f"%{search}%"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY ts DESC LIMIT :limit"
        params["limit"] = limit
        with _engine.connect() as conn:
            rows = conn.execute(text(sql), params).mappings().all()
            return [dict(r) for r in rows]
    except Exception as e:
        logger.warning("read_decisions failed: %s", e)
        return []


def row_counts() -> dict[str, int | None]:
    if not AVAILABLE or _engine is None:
        return {"predictions": None, "decisions": None, "bulk_predictions": None}
    try:
        with _engine.connect() as conn:
            p = conn.execute(text("SELECT count(*) FROM predictions")).scalar()
            d = conn.execute(text("SELECT count(*) FROM decisions")).scalar()
            b = conn.execute(text("SELECT count(*) FROM bulk_predictions")).scalar()
            return {"predictions": int(p), "decisions": int(d), "bulk_predictions": int(b)}
    except Exception as e:
        logger.warning("row_counts failed: %s", e)
        return {"predictions": None, "decisions": None, "bulk_predictions": None}


# =====================================================================
# Bulk prediction helpers (pages 5, 6, 7)
# =====================================================================

import json as _json
import math
import time


def _sanitize_for_json(obj):
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    return obj


def _isfraud_to_label(v) -> str | None:
    """Map an isFraud value (0/1/"0"/"1"/"fraud"/"legit") to user_label canonical string."""
    if v is None or v == "":
        return None
    try:
        n = int(float(v))
        return "fraud" if n == 1 else "legit" if n == 0 else None
    except (TypeError, ValueError):
        s = str(v).strip().lower()
        return "fraud" if s in ("fraud", "true", "yes") else "legit" if s in ("legit", "false", "no") else None


def create_bulk_job(filename: str, rows: list[dict[str, Any]]) -> int | None:
    """Insert all rows as pending. Returns the new job_id, or None if DB unavailable.

    If a row has an `isFraud` column (IEEE-CIS convention), it is stripped from
    the request payload and promoted to user_label so Bulk Dashboard can compute
    accuracy + confusion matrix automatically.
    """
    if not AVAILABLE or _engine is None:
        return None
    job_id = int(time.time() * 1000)  # ms epoch as unique-enough job_id
    try:
        with _engine.begin() as conn:
            for idx, row in enumerate(rows):
                transaction_id = row.get("transaction_id")
                payload_row = {k: v for k, v in row.items() if k != "isFraud"}
                user_label = _isfraud_to_label(row.get("isFraud"))
                # PostgreSQL JSONB rejects literal NaN/Infinity tokens; coerce to null.
                clean_row = _sanitize_for_json(payload_row)
                conn.execute(
                    text("""
                    INSERT INTO bulk_predictions
                        (job_id, job_filename, row_idx, transaction_id,
                         request_payload, status, user_label, labeled_at)
                    VALUES
                        (:job_id, :filename, :idx, :transaction_id,
                         CAST(:payload AS JSONB), 'pending', :user_label,
                         CASE WHEN :user_label IS NULL THEN NULL ELSE NOW() END)
                    """),
                    {
                        "job_id": job_id,
                        "filename": filename,
                        "idx": idx,
                        "transaction_id": int(transaction_id) if transaction_id else None,
                        "payload": _json.dumps(clean_row, default=str, allow_nan=False),
                        "user_label": user_label,
                    },
                )
        return job_id
    except Exception as e:
        logger.warning("create_bulk_job failed: %s", e)
        return None


def get_pending_rows(job_id: int, limit: int = 100) -> list[dict[str, Any]]:
    """Fetch rows that haven't been processed yet, for resumable workflow."""
    if not AVAILABLE or _engine is None:
        return []
    try:
        with _engine.connect() as conn:
            rows = conn.execute(
                text("""
                SELECT id, row_idx, request_payload
                FROM bulk_predictions
                WHERE job_id = :job_id AND status = 'pending'
                ORDER BY row_idx LIMIT :limit
                """),
                {"job_id": job_id, "limit": limit},
            ).mappings().all()
            return [dict(r) for r in rows]
    except Exception as e:
        logger.warning("get_pending_rows failed: %s", e)
        return []


def update_bulk_result(
    row_id: int,
    *,
    fraud_proba: float | None,
    predicted_label: int | None,
    model: str | None,
    error: str | None = None,
) -> None:
    """Mark a row completed or failed."""
    if not AVAILABLE or _engine is None:
        return
    status = "failed" if error else "completed"
    try:
        with _engine.begin() as conn:
            conn.execute(
                text("""
                UPDATE bulk_predictions
                SET status = :status,
                    fraud_proba = :proba,
                    predicted_label = :label,
                    model = :model,
                    error = :error
                WHERE id = :id
                """),
                {
                    "id": row_id,
                    "status": status,
                    "proba": fraud_proba,
                    "label": predicted_label,
                    "model": model,
                    "error": error,
                },
            )
    except Exception as e:
        logger.warning("update_bulk_result failed: %s", e)


def list_bulk_jobs() -> list[dict[str, Any]]:
    """Summary per job for dropdown selectors."""
    if not AVAILABLE or _engine is None:
        return []
    try:
        with _engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT
                    job_id,
                    MAX(job_filename) AS filename,
                    MIN(ts) AS started_at,
                    COUNT(*) AS total,
                    COUNT(*) FILTER (WHERE status='pending') AS pending,
                    COUNT(*) FILTER (WHERE status='completed') AS completed,
                    COUNT(*) FILTER (WHERE status='failed') AS failed,
                    COUNT(*) FILTER (WHERE user_label IS NOT NULL) AS labeled
                FROM bulk_predictions
                GROUP BY job_id
                ORDER BY started_at DESC
                LIMIT 50
            """)).mappings().all()
            return [dict(r) for r in rows]
    except Exception as e:
        logger.warning("list_bulk_jobs failed: %s", e)
        return []


def read_bulk_results(
    job_id: int,
    status_filter: list[str] | None = None,
    label_filter: list[str] | None = None,
    proba_min: float | None = None,
    proba_max: float | None = None,
    limit: int = 500,
) -> list[dict[str, Any]]:
    if not AVAILABLE or _engine is None:
        return []
    try:
        sql = """
            SELECT id, row_idx, ts, transaction_id, status, fraud_proba,
                   predicted_label, model, error, user_label, user_note,
                   request_payload
            FROM bulk_predictions
            WHERE job_id = :job_id
        """
        params: dict[str, Any] = {"job_id": job_id, "limit": limit}
        if status_filter:
            sql += " AND status = ANY(:status_filter)"
            params["status_filter"] = status_filter
        if label_filter:
            sql += " AND user_label = ANY(:label_filter)"
            params["label_filter"] = label_filter
        if proba_min is not None:
            sql += " AND fraud_proba >= :proba_min"
            params["proba_min"] = proba_min
        if proba_max is not None:
            sql += " AND fraud_proba <= :proba_max"
            params["proba_max"] = proba_max
        sql += " ORDER BY row_idx LIMIT :limit"
        with _engine.connect() as conn:
            rows = conn.execute(text(sql), params).mappings().all()
            return [dict(r) for r in rows]
    except Exception as e:
        logger.warning("read_bulk_results failed: %s", e)
        return []


_BULK_LABEL_TO_DECISION = {
    "fraud": "confirm_fraud",
    "legit": "approve_legit",
    "review": "need_more_info",
    "blocked": "blocked",
}


def label_bulk_rows(row_ids: list[int], user_label: str, user_note: str = "") -> int:
    """Apply a label to many rows at once. Returns count of rows updated.

    Also writes one row per labeled item into the `decisions` audit log with
    source='bulk' so Bulk Audit decisions show up in the Audit Log page.
    """
    if not AVAILABLE or _engine is None or not row_ids:
        return 0
    decision = _BULK_LABEL_TO_DECISION.get(user_label)
    try:
        with _engine.begin() as conn:
            result = conn.execute(
                text("""
                UPDATE bulk_predictions
                SET user_label = :label, user_note = :note, labeled_at = NOW()
                WHERE id = ANY(:ids)
                """),
                {"label": user_label, "note": user_note, "ids": row_ids},
            )
            n = int(result.rowcount or 0)

            if decision and n > 0:
                conn.execute(
                    text("""
                    INSERT INTO decisions
                        (transaction_id, source, decision, model_proba,
                         model_label, model, note)
                    SELECT transaction_id, 'bulk', :decision, fraud_proba,
                           predicted_label, model, :note
                    FROM bulk_predictions
                    WHERE id = ANY(:ids)
                    """),
                    {"decision": decision, "note": user_note, "ids": row_ids},
                )
            return n
    except Exception as e:
        logger.warning("label_bulk_rows failed: %s", e)
        return 0


# =====================================================================
# Continuous training (page 8)
# =====================================================================


def list_training_jobs(limit: int = 50) -> list[dict[str, Any]]:
    """Return recent training jobs for the Continuous Training page."""
    if not AVAILABLE or _engine is None:
        return []
    try:
        with _engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT id, ts, window_strategy, window_recent_days,
                       window_history_sample, n_new_labels, n_old_labels,
                       fraud_rate, model_version, model_path,
                       val_auc, train_seconds, status, promoted_at, notes
                FROM training_jobs
                ORDER BY ts DESC
                LIMIT :limit
            """), {"limit": limit}).mappings().all()
            return [dict(r) for r in rows]
    except Exception as e:
        logger.warning("list_training_jobs failed: %s", e)
        return []


def mark_training_job(job_id: int, *, status: str, promoted_by: str = "ui") -> bool:
    """Update status to 'promoted' or 'rejected'. Returns True if row updated."""
    if not AVAILABLE or _engine is None:
        return False
    try:
        with _engine.begin() as conn:
            result = conn.execute(text("""
                UPDATE training_jobs
                SET status = :status,
                    promoted_at = CASE WHEN :status = 'promoted' THEN NOW() ELSE promoted_at END,
                    promoted_by = CASE WHEN :status = 'promoted' THEN :by ELSE promoted_by END
                WHERE id = :id
            """), {"status": status, "by": promoted_by, "id": job_id})
            return (result.rowcount or 0) > 0
    except Exception as e:
        logger.warning("mark_training_job failed: %s", e)
        return False


def labeled_row_counts() -> dict[str, int | None]:
    """Sanity counts for the Continuous Training page UI."""
    if not AVAILABLE or _engine is None:
        return {"total_labeled": None, "fraud": None, "legit": None,
                "review": None, "blocked": None}
    try:
        with _engine.connect() as conn:
            stats = conn.execute(text("""
                SELECT
                    COUNT(*) FILTER (WHERE user_label IS NOT NULL) AS total_labeled,
                    COUNT(*) FILTER (WHERE user_label = 'fraud') AS fraud,
                    COUNT(*) FILTER (WHERE user_label = 'legit') AS legit,
                    COUNT(*) FILTER (WHERE user_label = 'review') AS review,
                    COUNT(*) FILTER (WHERE user_label = 'blocked') AS blocked
                FROM bulk_predictions
            """)).mappings().first()
            return {k: int(v) if v is not None else 0 for k, v in dict(stats).items()}
    except Exception as e:
        logger.warning("labeled_row_counts failed: %s", e)
        return {"total_labeled": None, "fraud": None, "legit": None,
                "review": None, "blocked": None}


def dashboard_metrics(job_id: int | None = None) -> dict[str, Any]:
    """Summary stats for dashboard. If job_id is None, returns global stats."""
    if not AVAILABLE or _engine is None:
        return {}
    try:
        with _engine.connect() as conn:
            where = "" if job_id is None else "WHERE job_id = :job_id"
            params = {} if job_id is None else {"job_id": job_id}
            summary = conn.execute(text(f"""
                SELECT
                    COUNT(*) AS total,
                    COUNT(*) FILTER (WHERE status='pending') AS pending,
                    COUNT(*) FILTER (WHERE status='completed') AS completed,
                    COUNT(*) FILTER (WHERE status='failed') AS failed,
                    COUNT(*) FILTER (WHERE predicted_label=1) AS predicted_fraud,
                    COUNT(*) FILTER (WHERE predicted_label=0) AS predicted_legit,
                    AVG(fraud_proba) AS avg_proba
                FROM bulk_predictions {where}
            """), params).mappings().first()

            by_label = conn.execute(text(f"""
                SELECT user_label, COUNT(*) AS n
                FROM bulk_predictions
                {where}
                GROUP BY user_label
            """), params).mappings().all()

            proba_buckets = conn.execute(text(f"""
                SELECT
                    WIDTH_BUCKET(fraud_proba, 0.0, 1.0, 10) AS bucket,
                    COUNT(*) AS n
                FROM bulk_predictions
                WHERE status='completed' AND fraud_proba IS NOT NULL
                {('AND job_id = :job_id' if job_id is not None else '')}
                GROUP BY bucket
                ORDER BY bucket
            """), params).mappings().all()

            return {
                "summary": dict(summary) if summary else {},
                "by_label": [dict(r) for r in by_label],
                "proba_buckets": [dict(r) for r in proba_buckets],
            }
    except Exception as e:
        logger.warning("dashboard_metrics failed: %s", e)
        return {}
