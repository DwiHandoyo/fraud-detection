"""PostgreSQL audit storage for predictions.

Graceful fallback: if DATABASE_URL is unset or connection fails at startup,
the module sets AVAILABLE=False and helpers become no-ops. Callers must not
crash — JSONL logging remains the source of truth.
"""
from __future__ import annotations

import json as _json
import logging
import os
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

logger = logging.getLogger("ml-service.db")

DATABASE_URL = os.getenv("DATABASE_URL")
_engine: Engine | None = None


def _init_engine() -> Engine | None:
    if not DATABASE_URL:
        logger.info("DATABASE_URL not set — DB logging disabled")
        return None
    try:
        eng = create_engine(DATABASE_URL, pool_pre_ping=True, pool_size=2, max_overflow=2)
        with eng.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("PostgreSQL connected")
        return eng
    except Exception as e:
        logger.warning("PostgreSQL connection failed: %s — falling back to JSONL only", e)
        return None


_engine = _init_engine()
AVAILABLE = _engine is not None


def insert_prediction(
    *,
    source: str,
    transaction_id: int | None,
    fraud_proba: float,
    predicted_label: int,
    model: str,
    model_version: str | None = None,
    request_payload: dict[str, Any] | None = None,
) -> int | None:
    """Insert a prediction row. Returns the new id, or None if DB unavailable."""
    if not AVAILABLE or _engine is None:
        return None
    try:
        with _engine.begin() as conn:
            result = conn.execute(
                text("""
                INSERT INTO predictions
                    (source, transaction_id, fraud_proba, predicted_label,
                     model, model_version, request_payload)
                VALUES
                    (:source, :transaction_id, :fraud_proba, :predicted_label,
                     :model, :model_version, CAST(:payload AS JSONB))
                RETURNING id
                """),
                {
                    "source": source,
                    "transaction_id": transaction_id,
                    "fraud_proba": float(fraud_proba),
                    "predicted_label": int(predicted_label),
                    "model": model,
                    "model_version": model_version,
                    "payload": _json.dumps(request_payload or {}, default=str),
                },
            )
            return int(result.scalar())
    except Exception as e:
        logger.warning("insert_prediction failed: %s", e)
        return None


def row_counts() -> dict[str, int | None]:
    if not AVAILABLE or _engine is None:
        return {"predictions": None, "decisions": None}
    try:
        with _engine.connect() as conn:
            p = conn.execute(text("SELECT count(*) FROM predictions")).scalar()
            d = conn.execute(text("SELECT count(*) FROM decisions")).scalar()
            return {"predictions": int(p), "decisions": int(d)}
    except Exception as e:
        logger.warning("row_counts failed: %s", e)
        return {"predictions": None, "decisions": None}
