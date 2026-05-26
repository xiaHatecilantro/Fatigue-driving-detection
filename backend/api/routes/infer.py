"""Image and video inference routes."""

from __future__ import annotations

from fastapi import APIRouter, File, Form, UploadFile

from backend.schemas.result import ImageInferenceResponse, VideoInferenceResponse
from backend.services.inference_service import inference_service


router = APIRouter(prefix="/api/infer", tags=["infer"])


@router.post(
    "/image",
    response_model=ImageInferenceResponse,
    summary="Run image inference",
    description="Upload one image and receive the unified fatigue/distraction result.",
)
async def infer_image(
    file: UploadFile = File(..., description="Input image file."),
    save_visualization: bool = Form(False, description="Whether to save an annotated output image."),
) -> ImageInferenceResponse:
    """Handle image inference request."""
    return await inference_service.infer_image(file=file, save_visualization=save_visualization)


@router.post(
    "/video",
    response_model=VideoInferenceResponse,
    summary="Run video inference",
    description="Upload one local video and run frame-by-frame fatigue/distraction analysis.",
)
async def infer_video(
    file: UploadFile = File(..., description="Input video file."),
    save_visualization: bool = Form(False, description="Whether to save an annotated output video."),
    include_frames: bool = Form(False, description="Whether to include per-frame unified results in the response."),
) -> VideoInferenceResponse:
    """Handle video inference request."""
    return await inference_service.infer_video(
        file=file,
        save_visualization=save_visualization,
        include_frames=include_frames,
    )
