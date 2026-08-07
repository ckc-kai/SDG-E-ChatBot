"""
FastAPI application entry point. 
- Creates the app
- configures CORS,
- mounts routers
Run with: uv run uvicorn main:app --reload (from backend/).
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import FRONTEND_ORIGINS, MODEL_WARMUP_ENABLED
from routers import ask, documents
from services.warmup_service import WarmupReport, warm_application_models


@asynccontextmanager
async def lifespan(app: FastAPI):
    if MODEL_WARMUP_ENABLED:
        app.state.model_warmup = warm_application_models(
            ask.get_generation_service()
        )
    else:
        app.state.model_warmup = WarmupReport(ready=False, errors=("disabled",))
    yield


app = FastAPI(title="SDG&E WMP Chatbot API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(FRONTEND_ORIGINS),
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ask.router)
app.include_router(documents.router)


@app.get("/api/health")
def health_check():
    report = getattr(app.state, "model_warmup", None)
    return {
        "status": "ok",
        "models_ready": bool(report and report.ready),
    }


@app.post("/api/warmup")
def warmup_models():
    """Refresh local model residency when a browser session starts."""
    report = warm_application_models(ask.get_generation_service())
    app.state.model_warmup = report
    return {
        "status": "ready" if report.ready else "degraded",
        "models_ready": report.ready,
    }
