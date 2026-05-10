"""Expectation suite for the train_identity dataset (batch validation).

Run on the parquet file in feature-store/feature_repo/data/identity.parquet
to catch schema drift, type errors, and unexpected values before training or
re-materialization.
"""
import great_expectations.expectations as gxe
from great_expectations.core import ExpectationSuite

REQUIRED_COLUMNS = [
    "transaction_id", "event_timestamp",
    "id_01", "id_02", "id_05", "id_11", "id_12", "id_14", "id_15", "id_16",
    "id_30", "id_31", "id_33", "id_35", "id_36", "id_37", "id_38",
    "DeviceType", "DeviceInfo",
]

CATEGORICAL_DOMAINS = {
    "DeviceType": ["mobile", "desktop"],
    "id_12": ["Found", "NotFound"],
    "id_15": ["Found", "New", "Unknown"],
    "id_16": ["Found", "NotFound"],
    "id_28": ["Found", "New"],
    "id_29": ["Found", "NotFound"],
    "id_35": ["T", "F"],
    "id_36": ["T", "F"],
    "id_37": ["T", "F"],
    "id_38": ["T", "F"],
}


def build() -> ExpectationSuite:
    suite = ExpectationSuite(name="identity_dataset")

    # Schema: all required columns present.
    for col in REQUIRED_COLUMNS:
        suite.add_expectation(gxe.ExpectColumnToExist(column=col))

    # Entity: transaction_id is unique positive integer.
    suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column="transaction_id"))
    suite.add_expectation(gxe.ExpectColumnValuesToBeUnique(column="transaction_id"))
    suite.add_expectation(gxe.ExpectColumnValuesToBeBetween(
        column="transaction_id", min_value=1, max_value=10_000_000_000,
    ))

    # id_11 is a percentage / rating score (observed range 0–100 in EDA).
    suite.add_expectation(gxe.ExpectColumnValuesToBeBetween(
        column="id_11", min_value=0, max_value=100, mostly=0.99,
    ))

    # id_14 is timezone offset in minutes (-12h to +14h).
    suite.add_expectation(gxe.ExpectColumnValuesToBeBetween(
        column="id_14", min_value=-720, max_value=840, mostly=0.99,
    ))

    # id_32 is screen color depth (16/24/32 bits).
    suite.add_expectation(gxe.ExpectColumnValuesToBeInSet(
        column="id_32", value_set=[16.0, 24.0, 32.0], mostly=0.99,
    ))

    # Categorical domains.
    for col, domain in CATEGORICAL_DOMAINS.items():
        suite.add_expectation(gxe.ExpectColumnValuesToBeInSet(
            column=col, value_set=domain, mostly=0.99,
        ))

    # Missing-rate guards: critical columns shouldn't be all-null.
    for col in ["id_30", "id_31", "DeviceType", "id_11", "id_14"]:
        suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(
            column=col, mostly=0.50,
        ))

    return suite
