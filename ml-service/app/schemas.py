"""Pydantic request/response schemas — model-agnostic.

Top-30 features by LightGBM gain are exposed explicitly for documentation;
any other feature can be passed via extra fields (extra="allow"). The adapter
pads/aligns to the model's expected feature set.
"""
from typing import Optional

from pydantic import BaseModel, Field

TOP_30_FEATURES = [
    "V258", "C14", "card1", "C13", "V201", "C1", "card2",
    "TransactionDT", "TransactionAmt", "V257", "addr1", "C11", "C8",
    "D2", "D15", "V317", "D1", "P_emaildomain", "V246", "card6",
    "C4", "D10", "V294", "dist1", "D4", "R_emaildomain", "card5",
    "C12", "C6", "C2",
]


class TransactionInput(BaseModel):
    """Request body for /predict and /explain.

    All fields optional. Adapter fills missing features with safe defaults
    (0.0 for numeric, 'null' for categorical) before inference.
    """

    model_config = {"extra": "allow"}

    TransactionAmt: Optional[float] = None
    TransactionDT: Optional[float] = None

    card1: Optional[float] = None
    card2: Optional[float] = None
    card5: Optional[float] = None
    card6: Optional[str] = None

    addr1: Optional[float] = None
    dist1: Optional[float] = None

    P_emaildomain: Optional[str] = None
    R_emaildomain: Optional[str] = None

    C1: Optional[float] = None
    C2: Optional[float] = None
    C4: Optional[float] = None
    C6: Optional[float] = None
    C8: Optional[float] = None
    C11: Optional[float] = None
    C12: Optional[float] = None
    C13: Optional[float] = None
    C14: Optional[float] = None

    D1: Optional[float] = None
    D2: Optional[float] = None
    D4: Optional[float] = None
    D10: Optional[float] = None
    D15: Optional[float] = None

    V201: Optional[float] = None
    V246: Optional[float] = None
    V257: Optional[float] = None
    V258: Optional[float] = None
    V294: Optional[float] = None
    V317: Optional[float] = None


class PredictByIdRequest(BaseModel):
    """Lookup features by transaction_id from the feature store, then predict."""
    transaction_id: int


class FeatureContribution(BaseModel):
    feature: str
    value: object  # could be int/float/str
    contribution: float


class PredictResponse(BaseModel):
    fraud_proba: float = Field(..., ge=0.0, le=1.0)
    predicted_label: int = Field(..., description="0 = legit, 1 = fraud")
    threshold: float
    model: str
    version: str


class ExplainResponse(BaseModel):
    fraud_proba: float
    contributions: list[FeatureContribution]
    model: str
    version: str


class HealthResponse(BaseModel):
    status: str
    predictor: str
    explainer: str
    n_predictor_features: int
    n_explainer_features: int
