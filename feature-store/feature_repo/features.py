"""Feast feature definitions for the fraud detection project.

One entity (transaction), one FeatureView covering all identity features dynamically,
one FeatureService that exposes the view.
"""
from datetime import timedelta
import pandas as pd  # <-- Imported pandas for dynamic schema generation

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

# === DYNAMIC SCHEMA GENERATION FROM PARQUET ===
def get_dynamic_schema(parquet_path):
    fields = []
    try:
        # Load only the first row to safely infer data types without heavy memory usage
        df_sample = pd.read_parquet(parquet_path, engine="pyarrow").iloc[:1]
        
        for col in df_sample.columns:
            # Skip primary join keys and timestamp columns
            if col in ["transaction_id", "event_timestamp", "created_timestamp"]:
                continue
                
            # Map pandas datatypes to Feast datatypes automatically
            dtype_str = str(df_sample[col].dtype)
            if "int" in dtype_str:
                feast_type = Int64
            elif "float" in dtype_str:
                feast_type = Float32
            else:
                feast_type = String
                
            fields.append(Field(name=col, dtype=feast_type))
    except Exception as e:
        print(f"[Warning] Failed to generate dynamic schema: {e}")
        # Provide an empty list fallback to prevent initialization crashes
        return []
    return fields

# Generate the schema field list instantly
dynamic_fields = get_dynamic_schema("data/identity.parquet")

# Feature view — Dynamically populates the schema list using dynamic_fields
identity_view = FeatureView(
    name="identity_view",
    entities=[transaction],
    ttl=timedelta(days=365 * 10),
    schema=dynamic_fields,  # <-- Injected dynamically here!
    source=identity_source,
    online=True,
)

# Feature service — what the inference layer asks for by name.
identity_service = FeatureService(
    name="identity_service",
    features=[identity_view],
)