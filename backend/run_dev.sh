#!/usr/bin/env bash

# Usage (from anywhere): bash backend/run_dev.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$SCRIPT_DIR"
export PYTHONPATH="$REPO_ROOT"
export SDGE_CONFIG_PATH="$REPO_ROOT/config/config.yaml"
export SDGE_EXCEL_CONTRACTS_PATH="$REPO_ROOT/config/excel_contracts.yaml"
uv run uvicorn main:app --reload
