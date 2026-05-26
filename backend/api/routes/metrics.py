"""Routes for training result visualization."""

from __future__ import annotations

from fastapi import APIRouter

from backend.schemas.result import TrainingMetricsResponse
from backend.services.metrics_service import metrics_service


router = APIRouter(prefix="/api/metrics", tags=["metrics"])


@router.get(
    "/training",
    response_model=TrainingMetricsResponse,
    summary="Get training metrics",
    description="Return baseline classifier metrics, best epoch, confusion matrix, and training curves.",
)
async def get_training_metrics() -> TrainingMetricsResponse:
    """Return cached training artifacts for frontend visualization."""
    return metrics_service.get_training_metrics()
