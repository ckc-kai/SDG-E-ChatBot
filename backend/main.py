"""
FastAPI application entry point. 
- Creates the app
- configures CORS,
- mounts routers
Run with: uv run uvicorn main:app --reload (from backend/).
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import FRONTEND_ORIGIN
from routers import ask, documents

app = FastAPI(title="SDG&E WMP Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ask.router)
app.include_router(documents.router)


@app.get("/api/health")
def health_check():
    return {"status": "ok"}