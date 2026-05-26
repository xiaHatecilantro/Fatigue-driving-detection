"""FastAPI application entrypoint for the driver monitoring backend."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes.health import router as health_router
from backend.api.routes.infer import router as infer_router
from backend.api.routes.metrics import router as metrics_router


app = FastAPI(
    title="Driver Fatigue and Distraction Detection API",
    version="0.1.0",
    description=(
        "Backend service for driver fatigue and distraction detection. "
        "The route layer only handles parameter parsing and response formatting, "
        "while the service layer owns inference orchestration. "
        "The current demo exposes offline image and video inference only."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(infer_router)
app.include_router(metrics_router)


@app.get("/", tags=["root"], summary="Root endpoint")
async def root() -> dict[str, str]:
    """Provide a minimal root response."""
    return {
        "message": "Driver monitoring backend is running in offline inference mode.",
        "docs": "/docs",
        "redoc": "/redoc",
    }
