"""Smoke tests for the Feast client wrapper."""
from pathlib import Path

import pytest

from app.feature_client import FeastClient

REPO = Path(__file__).resolve().parent.parent.parent / "feature-store" / "feature_repo"

# Known TransactionIDs from train_identity.csv head().
KNOWN_ID = 2987004
UNKNOWN_ID = 99_999_999


@pytest.fixture(scope="module")
def client():
    if not (REPO / "data" / "online_store.db").exists():
        pytest.skip("Online store not materialized — run materialize.sh first")
    return FeastClient(repo_path=REPO)


def test_known_id_returns_dataframe(client):
    df = client.get_features(KNOWN_ID)
    assert df is not None
    assert len(df) == 1
    assert df.shape[1] >= 30  # 33 features expected


def test_known_id_has_expected_values(client):
    df = client.get_features(KNOWN_ID).iloc[0]
    # From train_identity.csv row 1.
    assert df["id_30"] == "Android 7.0"
    assert df["id_31"] == "samsung browser 6.2"
    assert df["DeviceType"] == "mobile"
    assert df["id_14"] == -480.0


def test_unknown_id_returns_none(client):
    assert client.get_features(UNKNOWN_ID) is None
