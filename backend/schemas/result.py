"""Pydantic schemas for unified inference results and API responses."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


RiskLevel = Literal["normal", "mild", "moderate", "severe"]


class RuleInferenceResult(BaseModel):
    """Rule-based detector output before model fusion."""

    fatigue_score: float = Field(..., ge=0.0, le=100.0)
    distraction_score: float = Field(..., ge=0.0, le=100.0)
    risk_score: float = Field(..., ge=0.0, le=100.0)
    risk_level: RiskLevel
    status_labels: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    alarm_on: bool = False


class ModelInferenceSummary(BaseModel):
    """Classifier summary for one image or frame."""

    enabled: bool
    face_roi_used: bool = False
    predicted_label: str | None = None
    predicted_confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class FusionSummary(BaseModel):
    """Final fused service output summary."""

    fatigue_score: float = Field(..., ge=0.0, le=100.0)
    distraction_score: float = Field(..., ge=0.0, le=100.0)
    risk_level: RiskLevel
    alerts: list[str] = Field(default_factory=list)


class UnifiedInferenceResult(BaseModel):
    """Unified result schema shared by image, video, and realtime inference."""

    fatigue_score: float = Field(..., ge=0.0, le=100.0)
    distraction_score: float = Field(..., ge=0.0, le=100.0)
    risk_level: RiskLevel
    signals: dict[str, Any] = Field(default_factory=dict)
    model_probs: dict[str, float] = Field(default_factory=dict)
    alerts: list[str] = Field(default_factory=list)
    timestamp: str
    rule_result: RuleInferenceResult
    model_result: ModelInferenceSummary
    fusion_result: FusionSummary


class HealthResponse(BaseModel):
    """Health check response schema."""

    status: str
    service: str
    version: str


class ImageInferenceResponse(BaseModel):
    """Response schema for image inference."""

    status: str
    message: str
    result: UnifiedInferenceResult
    visualization_path: str | None = None


class VideoInferenceSummary(BaseModel):
    """Aggregate summary for offline video inference."""

    total_frames: int
    processed_frames: int
    failed_frames: int
    fps: float
    average_risk_score: float
    peak_risk_score: float
    peak_risk_level: RiskLevel
    event_count: int
    saved_visualization: bool


class VideoInferenceResponse(BaseModel):
    """Response schema for video inference."""

    status: str
    message: str
    summary: VideoInferenceSummary
    results: list[UnifiedInferenceResult] = Field(default_factory=list)
    events: list[dict[str, Any]] = Field(default_factory=list)
    visualization_path: str | None = None


class WebSocketReservationResponse(BaseModel):
    """Placeholder message for the reserved realtime websocket endpoint."""

    status: str
    message: str
    session_id: str | None = None


class TrainingMetrics(BaseModel):
    """Aggregate evaluation metrics for the classifier baseline."""

    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    precision_per_class: list[float] = Field(default_factory=list)
    recall_per_class: list[float] = Field(default_factory=list)
    f1_per_class: list[float] = Field(default_factory=list)
    confusion_matrix: list[list[int]] = Field(default_factory=list)


class TrainingMetricsResponse(BaseModel):
    """Read-only training result payload for frontend visualization."""

    model_name: str
    checkpoint: str
    best_epoch: int
    class_names: list[str] = Field(default_factory=list)
    metrics: TrainingMetrics
    train_curve: list[dict[str, Any]] = Field(default_factory=list)
