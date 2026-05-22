"""FastAPI app — model-agnostic. Endpoints route to adapters loaded from config.yaml."""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import cast

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .adapters import Explainer, Predictor, load
from .db import AVAILABLE as DB_AVAILABLE, insert_prediction
from .feature_client import FeastClient
from .schemas import (
    ExplainResponse,
    FeatureContribution,
    HealthResponse,
    PredictByIdRequest,
    PredictResponse,
    TransactionInput,
)
from .settings import load_config
from .validation import AVAILABLE as VALIDATION_AVAILABLE, get_validator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("ml-service")

cfg = load_config()
predictor = cast(Predictor, load(cfg["predictor"]))
explainer = cast(Explainer, load(cfg["explainer"]))
threshold = float(cfg.get("threshold", 0.5))

fs_cfg = cfg.get("feature_store", {})
feast_client: FeastClient | None = None
if fs_cfg.get("enabled"):
    repo = Path(fs_cfg["repo_path"])
    if not repo.is_absolute():
        repo = (Path(__file__).resolve().parent.parent / repo).resolve()
    feast_client = FeastClient(repo_path=repo, service_name=fs_cfg.get("service_name", "identity_service"))
    logger.info("feature store enabled, repo=%s", repo)
else:
    logger.info("feature store disabled")

val_cfg = cfg.get("validation", {})
validator = None
if val_cfg.get("enabled") and VALIDATION_AVAILABLE:
    validator = get_validator()
    logger.info("request validation enabled (great_expectations)")
else:
    logger.info("request validation disabled (enabled=%s, available=%s)",
                val_cfg.get("enabled"), VALIDATION_AVAILABLE)

logger.info("predictor=%s/%s features=%d", predictor.name, predictor.version, len(predictor.feature_names))
logger.info("explainer=%s/%s features=%d", explainer.name, explainer.version, len(explainer.feature_names))
logger.info("postgres audit logging: %s", "enabled" if DB_AVAILABLE else "disabled (JSONL only)")

# Structured prediction log — one JSON object per line. Read by Evidently
# for drift monitoring. Falls back to stdout if no LOG_FILE set.
PREDICTIONS_LOG_PATH = os.getenv("LOG_FILE")
predictions_logger = logging.getLogger("predictions")
predictions_logger.setLevel(logging.INFO)
predictions_logger.propagate = False
if PREDICTIONS_LOG_PATH:
    Path(PREDICTIONS_LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
    _h = logging.FileHandler(PREDICTIONS_LOG_PATH)
else:
    _h = logging.StreamHandler()
_h.setFormatter(logging.Formatter("%(message)s"))
predictions_logger.addHandler(_h)


def _log_prediction(
    *,
    source: str,
    transaction_id,
    fraud_proba: float,
    label: int,
    model_name: str,
    model_version: str | None = None,
    request_payload: dict | None = None,
) -> int | None:
    """Dual-write: JSONL (for Evidently) + Postgres (for Audit Log). Returns
    prediction_id from Postgres if available, else None."""
    predictions_logger.info(json.dumps({
        "ts": datetime.utcnow().isoformat() + "Z",
        "source": source,
        "transaction_id": transaction_id,
        "fraud_proba": round(fraud_proba, 6),
        "label": label,
        "model": model_name,
    }))
    return insert_prediction(
        source=source,
        transaction_id=transaction_id,
        fraud_proba=fraud_proba,
        predicted_label=label,
        model=model_name,
        model_version=model_version,
        request_payload=request_payload,
    )

app = FastAPI(
    title="Fraud Detection Service",
    version="1.0.0",
    description="Model-agnostic inference service. Swap models via config.yaml.",
)


def _to_df(req: TransactionInput) -> pd.DataFrame:
    """Convert a single Pydantic input to a 1-row DataFrame, including extras."""
    return pd.DataFrame([req.model_dump(exclude_none=False)])


def _validate_or_422(req: TransactionInput) -> None:
    if validator is None:
        return
    payload = {k: v for k, v in req.model_dump().items() if v is not None}
    if not payload:
        return
    ok, errors = validator.validate(payload)
    if not ok:
        raise HTTPException(status_code=422, detail={"validation_errors": errors})


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        predictor=f"{predictor.name}/{predictor.version}",
        explainer=f"{explainer.name}/{explainer.version}",
        n_predictor_features=len(predictor.feature_names),
        n_explainer_features=len(explainer.feature_names),
    )


@app.post("/predict", response_model=PredictResponse)
def predict(req: TransactionInput) -> PredictResponse:
    _validate_or_422(req)
    X = _to_df(req)
    proba = predictor.predict_proba(X)[0]
    label = int(proba >= threshold)
    _log_prediction(
        source="predict", transaction_id=None, fraud_proba=proba, label=label,
        model_name=predictor.name, model_version=predictor.version,
        request_payload=req.model_dump(exclude_none=True),
    )
    return PredictResponse(
        fraud_proba=proba,
        predicted_label=label,
        threshold=threshold,
        model=predictor.name,
        version=predictor.version,
    )


@app.post("/explain", response_model=ExplainResponse)
def explain(req: TransactionInput) -> ExplainResponse:
    _validate_or_422(req)
    X = _to_df(req)
    proba = explainer.predict_proba(X)[0]
    contribs_raw = explainer.explain(X)[0]
    return ExplainResponse(
        fraud_proba=proba,
        contributions=[FeatureContribution(**c) for c in contribs_raw],
        model=explainer.name,
        version=explainer.version,
    )


def _lookup_or_404(transaction_id: int) -> pd.DataFrame:
    if feast_client is None:
        raise HTTPException(503, "Feature store not configured")
    X = feast_client.get_features(transaction_id)
    if X is None:
        raise HTTPException(404, f"transaction_id {transaction_id} not found in feature store")
    return X


@app.post("/predict_by_id", response_model=PredictResponse)
def predict_by_id(req: PredictByIdRequest) -> PredictResponse:
    X = _lookup_or_404(req.transaction_id)
    proba = predictor.predict_proba(X)[0]
    label = int(proba >= threshold)
    _log_prediction(
        source="predict_by_id", transaction_id=req.transaction_id, fraud_proba=proba,
        label=label, model_name=predictor.name, model_version=predictor.version,
    )
    return PredictResponse(
        fraud_proba=proba,
        predicted_label=label,
        threshold=threshold,
        model=predictor.name,
        version=predictor.version,
    )


@app.post("/explain_by_id", response_model=ExplainResponse)
def explain_by_id(req: PredictByIdRequest) -> ExplainResponse:
    X = _lookup_or_404(req.transaction_id)
    proba = explainer.predict_proba(X)[0]
    contribs_raw = explainer.explain(X)[0]
    return ExplainResponse(
        fraud_proba=proba,
        contributions=[FeatureContribution(**c) for c in contribs_raw],
        model=explainer.name,
        version=explainer.version,
    )


# =====================================================================
# Admin — continuous training trigger
# =====================================================================

class RetrainRequest(BaseModel):
    recent_days: int = 30
    history_sample_rate: float = 0.20
    min_new_labels: int = 20
    quick: bool = True
    notes: str = ""


class RetrainResponse(BaseModel):
    status: str               # 'success' | 'error'
    returncode: int
    stdout: str
    stderr: str


RETRAIN_SCRIPT = Path("/playground/lgbm/retrain_from_feedback.py")


@app.post("/admin/retrain", response_model=RetrainResponse)
def admin_retrain(req: RetrainRequest) -> RetrainResponse:
    """Trigger retrain script as subprocess. Synchronous (blocks for 1–10 min).

    Mounted in docker-compose.yml:
      ./playground:/playground:ro      (read-only, holds the script)
      ./ml-service/models:/app/models  (read-write, retrain writes new pkl here)
    """
    if not RETRAIN_SCRIPT.exists():
        raise HTTPException(
            500,
            f"Retrain script not found at {RETRAIN_SCRIPT}. "
            "Mount `./playground:/playground:ro` in docker-compose.yml.",
        )

    cmd = [
        sys.executable, str(RETRAIN_SCRIPT),
        "--recent-days", str(req.recent_days),
        "--history-sample-rate", str(req.history_sample_rate),
        "--min-new-labels", str(req.min_new_labels),
        "--output-dir", "/app/models",
    ]
    if req.quick:
        cmd.append("--quick")
    if req.notes:
        cmd.extend(["--notes", req.notes])

    logger.info("Starting retrain: %s", " ".join(cmd))
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=900,
            env={**os.environ},
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(504, "Retrain timed out (>15 minutes).")

    status = "success" if result.returncode == 0 else "error"
    logger.info("Retrain finished with code %d", result.returncode)
    return RetrainResponse(
        status=status,
        returncode=result.returncode,
        stdout=result.stdout[-5000:],  # cap to avoid huge HTTP body
        stderr=result.stderr[-5000:],
    )
