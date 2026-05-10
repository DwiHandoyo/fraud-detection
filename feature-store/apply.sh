#!/usr/bin/env bash
# Register feature definitions with Feast (writes data/registry.db).
set -euo pipefail
cd "$(dirname "$0")/feature_repo"
feast apply
