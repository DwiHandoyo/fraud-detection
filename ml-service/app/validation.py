"""Bridge to the data-validation package.

We don't pip-install data-validation as a library; ml-service simply adds the
sibling folder to sys.path and reuses the request validator.
"""
from __future__ import annotations

import sys
from pathlib import Path

DATA_VAL_DIR = Path(__file__).resolve().parent.parent.parent / "data-validation"
if DATA_VAL_DIR.exists() and str(DATA_VAL_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_VAL_DIR))

try:
    from validate_request import get_validator  # noqa: E402,F401
    AVAILABLE = True
except Exception:  # noqa: BLE001
    AVAILABLE = False

    def get_validator():  # type: ignore[no-redef]
        raise RuntimeError(
            "data-validation package not importable from "
            f"{DATA_VAL_DIR}. Install it or disable validation in config.yaml."
        )
