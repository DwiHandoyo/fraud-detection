"""Model adapters — wrap concrete ML models behind a common interface.

Add a new model:
  1. Subclass nothing — just match the Predictor / Explainer Protocol shape.
  2. Register in ADAPTERS dict at the bottom.
  3. Reference by `type:` in config.yaml.
"""
from __future__ import annotations

from typing import Protocol

import joblib
import numpy as np
import pandas as pd


class Predictor(Protocol):
    name: str
    version: str
    feature_names: list[str]

    def predict_proba(self, X: pd.DataFrame) -> list[float]: ...


class Explainer(Protocol):
    name: str
    version: str
    feature_names: list[str]

    def predict_proba(self, X: pd.DataFrame) -> list[float]: ...
    def explain(self, X: pd.DataFrame) -> list[list[dict]]: ...


def _align(X: pd.DataFrame, expected: list[str]) -> pd.DataFrame:
    """Reindex to expected columns, fill missing with safe defaults.

    Numeric NaN -> 0.0 (tree models handle 0 as neutral). String NaN -> 'null'.
    """
    aligned = X.reindex(columns=expected)
    for col in aligned.columns:
        if aligned[col].dtype == "object":
            aligned[col] = aligned[col].fillna("null")
        else:
            aligned[col] = aligned[col].fillna(0.0)
    return aligned


def _factorize_objects(X: pd.DataFrame) -> pd.DataFrame:
    """Stop-gap: factorize remaining object columns to int.

    Used when no fitted preprocessor is available for a categorical column
    (e.g. LGBM's full feature set when preprocessor was fit on identity only).
    """
    out = X.copy()
    for col in out.select_dtypes(include=["object"]).columns:
        out[col] = pd.factorize(out[col])[0]
    return out


class LGBMAdapter:
    """Adapter for sklearn-API LightGBM classifier."""

    def __init__(self, path: str, version: str = "1.0"):
        self.model = joblib.load(path)
        self.feature_names = list(self.model.feature_name_)
        self.name = "lgbm"
        self.version = version

    def predict_proba(self, X: pd.DataFrame) -> list[float]:
        X_aligned = _align(X, self.feature_names)
        X_aligned = _factorize_objects(X_aligned)
        return self.model.predict_proba(X_aligned)[:, 1].tolist()


class EBMAdapter:
    """Adapter for InterpretML ExplainableBoostingClassifier (with explanations)."""

    def __init__(self, path: str, version: str = "1.0"):
        bundle = joblib.load(path)
        self.model = bundle["model"]
        self.feature_names = list(bundle["feature_names"])
        self.name = "ebm"
        self.version = version

    def predict_proba(self, X: pd.DataFrame) -> list[float]:
        X_aligned = _align(X, self.feature_names)
        return self.model.predict_proba(X_aligned)[:, 1].tolist()

    def explain(self, X: pd.DataFrame) -> list[list[dict]]:
        X_aligned = _align(X, self.feature_names)
        local = self.model.explain_local(X_aligned)
        results = []
        for i in range(len(X_aligned)):
            data = local.data(i)
            row = []
            for name, score, value in zip(data["names"], data["scores"], data["values"]):
                row.append({
                    "feature": str(name),
                    "value": _coerce(value),
                    "contribution": float(score),
                })
            row.sort(key=lambda r: abs(r["contribution"]), reverse=True)
            results.append(row)
        return results


def _coerce(v):
    """Make a value JSON-serializable."""
    if isinstance(v, (np.integer, np.floating)):
        return v.item()
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
        return None
    return v


# Registry — add new model types here.
ADAPTERS = {
    "lgbm": LGBMAdapter,
    "ebm": EBMAdapter,
}


def load(spec: dict):
    """Instantiate an adapter from a config dict like {'type': 'lgbm', 'path': ..., 'version': '1.0'}."""
    cls = ADAPTERS.get(spec["type"])
    if cls is None:
        raise ValueError(f"Unknown adapter type: {spec['type']}. Available: {list(ADAPTERS)}")
    return cls(path=spec["path"], version=spec.get("version", "1.0"))
