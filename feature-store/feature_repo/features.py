"""Feast feature definitions for the fraud detection project.

One entity (transaction), one FeatureView covering all 33 identity features,
one FeatureService that exposes the view.
"""
from datetime import timedelta

from feast import Entity, FeatureService, FeatureView, Field, FileSource
from feast.types import Float32, Int64, String

# Entity — the join key Feast will use at lookup time.
transaction = Entity(
    name="transaction",
    join_keys=["transaction_id"],
    description="A single payment transaction (TransactionID from IEEE-CIS).",
)

# Source — the parquet seeded from train_identity.csv.
identity_source = FileSource(
    name="identity_source",
    path="data/identity.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

# Feature view — 33 identity features (numeric float32 + categorical string).
# Numeric ID fields kept as Float32 for compactness.
identity_view = FeatureView(
    name="identity_view",
    entities=[transaction],
    ttl=timedelta(days=365 * 10),
    schema=[
        Field(name="id_01", dtype=Float32),
        Field(name="id_02", dtype=Float32),
        Field(name="id_03", dtype=Float32),
        Field(name="id_04", dtype=Float32),
        Field(name="id_05", dtype=Float32),
        Field(name="id_06", dtype=Float32),
        Field(name="id_07", dtype=Float32),
        Field(name="id_08", dtype=Float32),
        Field(name="id_09", dtype=Float32),
        Field(name="id_10", dtype=Float32),
        Field(name="id_11", dtype=Float32),
        Field(name="id_12", dtype=String),
        Field(name="id_13", dtype=Float32),
        Field(name="id_14", dtype=Float32),
        Field(name="id_15", dtype=String),
        Field(name="id_16", dtype=String),
        Field(name="id_17", dtype=Float32),
        Field(name="id_18", dtype=Float32),
        Field(name="id_19", dtype=Float32),
        Field(name="id_20", dtype=Float32),
        Field(name="id_28", dtype=String),
        Field(name="id_29", dtype=String),
        Field(name="id_30", dtype=String),
        Field(name="id_31", dtype=String),
        Field(name="id_32", dtype=Float32),
        Field(name="id_33", dtype=String),
        Field(name="id_34", dtype=String),
        Field(name="id_35", dtype=String),
        Field(name="id_36", dtype=String),
        Field(name="id_37", dtype=String),
        Field(name="id_38", dtype=String),
        Field(name="DeviceType", dtype=String),
        Field(name="DeviceInfo", dtype=String),
    ],
    source=identity_source,
    online=True,
)

# Feature service — what the inference layer asks for by name.
identity_service = FeatureService(
    name="identity_service",
    features=[identity_view],
)
