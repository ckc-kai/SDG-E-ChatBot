#!/usr/bin/env bash
#
# Usage (from anywhere): bash backend/run_dev.sh
set -e
cd "$(dirname "$0")"
REPO_ROOT="$(cd .. && pwd)"

export PYTHONPATH="$REPO_ROOT"
export SDGE_CONFIG_PATH="$REPO_ROOT/config/config.yaml"
export SDGE_EXCEL_CONTRACTS_PATH="$REPO_ROOT/config/excel_contracts.yaml"
export PYTHONUTF8=1
export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1

uv run uvicorn main:app --reload