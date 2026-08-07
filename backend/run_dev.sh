#!/usr/bin/env bash

# Usage (from anywhere): bash backend/run_dev.sh
set -e
cd "$(dirname "$0")"
export PYTHONPATH=..
export SDGE_CONFIG_PATH="$(cd .. && pwd)/config/config.yaml"
uv run uvicorn main:app --reload