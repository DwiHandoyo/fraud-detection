"""Batch-validate the identity.parquet against the identity_dataset suite.

Outputs:
  reports/dataset_validation.json    machine-readable result
  gx/uncommitted/data_docs/...       Data Docs HTML site

Usage:
    python validate_dataset.py [PATH_TO_PARQUET]

Default path: ../feature-store/feature_repo/data/identity.parquet
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import great_expectations as gx
import pandas as pd

from suites.identity_dataset import build as build_suite

HERE = Path(__file__).resolve().parent
DEFAULT_PARQUET = HERE.parent / "feature-store" / "feature_repo" / "data" / "identity.parquet"
GX_DIR = HERE / "gx"
REPORTS = HERE / "reports"


def main() -> int:
    parquet = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PARQUET
    print(f"[validate] reading {parquet}")
    df = pd.read_parquet(parquet)
    print(f"[validate] shape: {df.shape}")

    GX_DIR.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)

    context = gx.get_context(mode="file", project_root_dir=str(HERE))

    data_source = context.data_sources.add_or_update_pandas("identity_pandas")
    asset = data_source.add_dataframe_asset(name="identity_asset")
    batch_def = asset.add_batch_definition_whole_dataframe("identity_batch")

    suite = build_suite()
    try:
        context.suites.delete(name="identity_dataset")
    except Exception:
        pass
    suite = context.suites.add(suite)

    vd = context.validation_definitions.add_or_update(
        gx.ValidationDefinition(name="identity_validation", data=batch_def, suite=suite)
    )

    result = vd.run(batch_parameters={"dataframe": df})

    summary = {
        "success": result.success,
        "evaluated_expectations": result.statistics["evaluated_expectations"],
        "successful_expectations": result.statistics["successful_expectations"],
        "unsuccessful_expectations": result.statistics["unsuccessful_expectations"],
        "success_percent": result.statistics["success_percent"],
    }
    failed = [
        {
            "expectation": r.expectation_config.type,
            "column": r.expectation_config.kwargs.get("column"),
            "details": (r.result or {}).get("partial_unexpected_list", [])[:5],
        }
        for r in result.results if not r.success
    ]
    summary["failed_expectations"] = failed

    out_json = REPORTS / "dataset_validation.json"
    out_json.write_text(json.dumps(summary, indent=2, default=str))
    print(f"[validate] wrote {out_json}")

    # Generate Data Docs HTML site.
    if "local" not in context.list_data_docs_sites():
        context.add_data_docs_site(
            site_name="local",
            site_config={
                "class_name": "SiteBuilder",
                "store_backend": {
                    "class_name": "TupleFilesystemStoreBackend",
                    "base_directory": str(REPORTS / "data_docs"),
                },
                "site_index_builder": {"class_name": "DefaultSiteIndexBuilder"},
            },
        )
    context.build_data_docs(site_names=["local"])
    print(f"[validate] HTML docs at {REPORTS / 'data_docs' / 'index.html'}")

    print(f"\n[result] success={result.success}  "
          f"{summary['successful_expectations']}/{summary['evaluated_expectations']} passed "
          f"({summary['success_percent']:.1f}%)")
    if not result.success:
        print(f"[result] {len(failed)} failed:")
        for f in failed:
            print(f"  - {f['expectation']} on {f['column']}: {f['details']}")
    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
