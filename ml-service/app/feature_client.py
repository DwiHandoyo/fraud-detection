"""Feast online-store client wrapper.

Loads the FeatureStore at startup, exposes one method that maps a
transaction_id -> single-row DataFrame ready for adapter consumption.
Returns None when the entity is not found (all Feast values come back null).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from feast import FeatureStore

# Feast returns None for NaN values. With a single-row lookup pandas infers the
# column dtype as `object`, which EBM rejects ("indicated as continuous, but
# has non-numeric data"). We coerce known-numeric columns to float64 explicitly.
# Mirrors the Float32 fields in feature-store/feature_repo/features.py.
_NUMERIC_FEATURES = {
    "id_01", "id_02", "id_03", "id_04", "id_05", "id_06", "id_07", "id_08",
    "id_09", "id_10", "id_11", "id_13", "id_14", "id_17", "id_18", "id_19",
    "id_20", "id_32",
}


class FeastClient:
    def __init__(self, repo_path: str | Path, service_name: str = "identity_service"):
        self.repo_path = Path(repo_path)
        self.store = FeatureStore(repo_path=str(self.repo_path))
        self.service = self.store.get_feature_service(service_name)

    def get_features(self, transaction_id: int) -> pd.DataFrame | None:
        """Return a 1-row DataFrame for `transaction_id`, or None if unknown."""
        result = self.store.get_online_features(
            features=self.service,
            entity_rows=[{"transaction_id": int(transaction_id)}],
        ).to_dict()
        df = pd.DataFrame(result)

        feature_cols = [c for c in df.columns if c != "transaction_id"]
        if df[feature_cols].isna().all().all():
            return None

        df = df.drop(columns=["transaction_id"])
        for col in df.columns:
            if col in _NUMERIC_FEATURES:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
        return df
