"""FastAPI app — model-agnostic. Endpoints route to adapters loaded from config.yaml."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import cast

import pandas as pd
from fastapi import FastAPI, HTTPException

from .adapters import Explainer, Predictor, load
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

logger.info("predictor=%s/%s features=%d", predictor.name, predictor.version, len(predictor.feature_names))
logger.info("explainer=%s/%s features=%d", explainer.name, explainer.version, len(explainer.feature_names))

app = FastAPI(
    title="Fraud Detection Service",
    version="1.0.0",
    description="Model-agnostic inference service. Swap models via config.yaml.",
)


def _to_df(req: TransactionInput) -> pd.DataFrame:
    """Convert a single Pydantic input to a 1-row DataFrame, including extras."""
    return pd.DataFrame([req.model_dump(exclude_none=False)])


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
    X = _to_df(req)
    proba = predictor.predict_proba(X)[0]
    return PredictResponse(
        fraud_proba=proba,
        predicted_label=int(proba >= threshold),
        threshold=threshold,
        model=predictor.name,
        version=predictor.version,
    )


@app.post("/explain", response_model=ExplainResponse)
def explain(req: TransactionInput) -> ExplainResponse:
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
    return PredictResponse(
        fraud_proba=proba,
        predicted_label=int(proba >= threshold),
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
