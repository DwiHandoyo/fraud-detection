#!/usr/bin/env bash
# Register feature definitions with Feast (writes data/registry.db).
set -euo pipefail
cd "$(dirname "$0")/feature_repo"

# Hapus registry lama jika ada, untuk menghindari error absolute path saat berpindah dari host ke Docker
if [ -f "data/registry.db" ]; then
    rm data/registry.db
fi

feast apply
