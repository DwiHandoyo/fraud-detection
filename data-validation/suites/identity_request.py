"""Expectation suite for a single inference request payload.

Used as middleware: validates the JSON body of /predict before it reaches the
model. Catches obvious payload errors (negative amount, invalid timezone, etc.)
and returns a 422 response.
"""
import great_expectations.expectations as gxe
from great_expectations.core import ExpectationSuite

from .identity_dataset import CATEGORICAL_DOMAINS


def build() -> ExpectationSuite:
    suite = ExpectationSuite(name="identity_request")

    # TransactionAmt must be positive when provided.
    suite.add_expectation(gxe.ExpectColumnValuesToBeBetween(
        column="TransactionAmt", min_value=0, max_value=100_000,
    ))

    # Timezone offset.
    suite.add_expectation(gxe.ExpectColumnValuesToBeBetween(
        column="id_14", min_value=-720, max_value=840,
    ))

    # Rating score percentage.
    suite.add_expectation(gxe.ExpectColumnValuesToBeBetween(
        column="id_11", min_value=0, max_value=100,
    ))

    # Categorical domains (allow null — many fields are optional).
    for col, domain in CATEGORICAL_DOMAINS.items():
        suite.add_expectation(gxe.ExpectColumnValuesToBeInSet(
            column=col, value_set=domain,
        ))

    return suite
