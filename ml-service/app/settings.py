"""Config loading. Single YAML at the service root."""
from __future__ import annotations

import os
from pathlib import Path

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config.yaml"


def load_config(path: str | Path | None = None) -> dict:
    """Load config.yaml. Honors CONFIG_PATH env var for container overrides."""
    cfg_path = Path(path or os.getenv("CONFIG_PATH", DEFAULT_CONFIG))
    with cfg_path.open() as f:
        cfg = yaml.safe_load(f)

    base = cfg_path.parent
    for key in ("predictor", "explainer", "preprocessor"):
        if key in cfg and "path" in cfg[key]:
            p = Path(cfg[key]["path"])
            if not p.is_absolute():
                cfg[key]["path"] = str((base / p).resolve())
    return cfg
