"""Health check route."""

from __future__ import annotations

from fastapi import APIRouter

from backend.schemas.result import HealthResponse


router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse, summary="Health check")
async def health_check() -> HealthResponse:
    """Return service health status."""
    return HealthResponse(status="ok", service="driver-monitoring-backend", version="0.1.0")
