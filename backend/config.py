"""
Location: backend/config.py
Role: backend-specific settings that are NOT already covered by the
shared config/config.yaml (which Member 2 owns -- chunk sizes, embedding
model, retrieval_top_k, etc. all stay there, we don't duplicate them here).
This file is only for things specific to the FastAPI app itself.
"""
import os

# Comma-separated so local development works with either common loopback host.
FRONTEND_ORIGINS = tuple(
    origin.strip()
    for origin in os.getenv(
        "FRONTEND_ORIGIN",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if origin.strip()
)

MODEL_WARMUP_ENABLED = os.getenv("MODEL_WARMUP_ENABLED", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

# default model here currently needs to be updated to the final model... 
DEFAULT_ANSWER_MODEL = os.getenv("DEFAULT_ANSWER_MODEL", "claude-sonnet-4-6")
