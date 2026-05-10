#!/usr/bin/env bash
# Populate the SQLite online store from the parquet offline store.
# Pass an end date as $1 (defaults to today).
set -euo pipefail
cd "$(dirname "$0")/feature_repo"
END_DATE="${1:-$(date -u +%Y-%m-%d)T00:00:00}"
feast materialize-incremental "$END_DATE"
