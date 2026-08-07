"""
Location: backend/config.py
Role: backend-specific settings that are NOT already covered by the
shared config/config.yaml (which Member 2 owns -- chunk sizes, embedding
model, retrieval_top_k, etc. all stay there, we don't duplicate them here).
This file is only for things specific to the FastAPI app itself.
"""
import os

# url needs to be updated to actual website hosting site.... 
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")

# default model here currently needs to be updated to the final model... 
DEFAULT_ANSWER_MODEL = os.getenv("DEFAULT_ANSWER_MODEL", "claude-sonnet-4-6")