"""Runtime validator for a single inference request payload.

Used by ml-service as a pre-check before calling the model. The expectation
suite is built once per process; each call is a tiny in-memory pandas validate.

Usage:
    from validate_request import RequestValidator
    v = RequestValidator()
    ok, errors = v.validate({"TransactionAmt": 250, "id_14": -300})
"""
from __future__ import annotations

import warnings
from pathlib import Path

import great_expectations as gx
import pandas as pd

from suites.identity_request import build as build_suite

HERE = Path(__file__).resolve().parent


class RequestValidator:
    def __init__(self):
        # Suppress GX warnings — they're noisy in request paths.
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", module="great_expectations")
        self._context = gx.get_context(mode="ephemeral")
        ds = self._context.data_sources.add_pandas("request_pandas")
        asset = ds.add_dataframe_asset(name="request_asset")
        self._batch_def = asset.add_batch_definition_whole_dataframe("request_batch")
        self._suite = self._context.suites.add(build_suite())
        self._vd = self._context.validation_definitions.add(
            gx.ValidationDefinition(name="request_validation",
                                    data=self._batch_def, suite=self._suite)
        )

    def validate(self, payload: dict) -> tuple[bool, list[dict]]:
        """Validate a single payload. Returns (ok, list of failed expectations)."""
        # Only validate keys that exist in the payload — request fields are optional.
        df = pd.DataFrame([payload])
        try:
            result = self._vd.run(batch_parameters={"dataframe": df})
        except Exception as e:
            return False, [{"error": f"validation crashed: {e}"}]

        if result.success:
            return True, []

        failed = []
        for r in result.results:
            if r.success:
                continue
            col = r.expectation_config.kwargs.get("column")
            # If the column isn't in the payload, skip — fields are optional.
            if col and col not in payload:
                continue
            failed.append({
                "expectation": r.expectation_config.type,
                "column": col,
                "observed": (r.result or {}).get("partial_unexpected_list", []),
                "expected_kwargs": {
                    k: v for k, v in r.expectation_config.kwargs.items()
                    if k != "column"
                },
            })
        return (len(failed) == 0), failed


_singleton: RequestValidator | None = None


def get_validator() -> RequestValidator:
    """Lazy singleton — keeps GX context warm across requests."""
    global _singleton
    if _singleton is None:
        _singleton = RequestValidator()
    return _singleton
