"""Smoke tests for adapters. Loads real .pkl artifacts."""
from pathlib import Path

import pandas as pd
import pytest

from app.adapters import EBMAdapter, LGBMAdapter, load

MODELS = Path(__file__).resolve().parent.parent / "models"


def _sample_row() -> pd.DataFrame:
    return pd.DataFrame([{
        "TransactionAmt": 250.0,
        "card4": "visa",
        "P_emaildomain": "gmail.com",
        "id_31": "chrome 62.0",
        "id_30": "Windows 10",
        "id_14": -300.0,
        "DeviceType": "desktop",
    }])


def test_lgbm_adapter_loads():
    adapter = LGBMAdapter(path=str(MODELS / "lgbm_best.pkl"))
    assert adapter.name == "lgbm"
    assert len(adapter.feature_names) == 107


def test_lgbm_predict_proba_returns_float_in_unit_interval():
    adapter = LGBMAdapter(path=str(MODELS / "lgbm_best.pkl"))
    proba = adapter.predict_proba(_sample_row())
    assert len(proba) == 1
    assert 0.0 <= proba[0] <= 1.0


def test_lgbm_explain_returns_per_feature_contributions():
    adapter = LGBMAdapter(path=str(MODELS / "lgbm_best.pkl"))
    explanations = adapter.explain(_sample_row())
    assert len(explanations) == 1
    contribs = explanations[0]
    assert all("feature" in c and "contribution" in c for c in contribs)
    # Sorted by absolute contribution descending.
    abs_scores = [abs(c["contribution"]) for c in contribs]
    assert abs_scores == sorted(abs_scores, reverse=True)
    # Same number of contributions as features.
    assert len(contribs) == len(adapter.feature_names)


def test_ebm_adapter_loads():
    adapter = EBMAdapter(path=str(MODELS / "ebm.pkl"))
    assert adapter.name == "ebm"
    assert len(adapter.feature_names) == 25


def test_ebm_predict_proba_returns_float_in_unit_interval():
    adapter = EBMAdapter(path=str(MODELS / "ebm.pkl"))
    proba = adapter.predict_proba(_sample_row())
    assert 0.0 <= proba[0] <= 1.0


def test_ebm_explain_returns_per_feature_contributions():
    adapter = EBMAdapter(path=str(MODELS / "ebm.pkl"))
    explanations = adapter.explain(_sample_row())
    assert len(explanations) == 1
    contribs = explanations[0]
    assert all("feature" in c and "contribution" in c for c in contribs)
    # Sorted by absolute contribution descending.
    abs_scores = [abs(c["contribution"]) for c in contribs]
    assert abs_scores == sorted(abs_scores, reverse=True)


def test_load_dispatches_by_type():
    spec = {"type": "ebm", "path": str(MODELS / "ebm.pkl"), "version": "test"}
    adapter = load(spec)
    assert adapter.name == "ebm"
    assert adapter.version == "test"


def test_load_unknown_type_raises():
    with pytest.raises(ValueError):
        load({"type": "nonexistent", "path": "x"})
