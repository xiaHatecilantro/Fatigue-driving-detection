"""Realtime websocket route."""

from __future__ import annotations

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.services.inference_service import inference_service

router = APIRouter(tags=["ws"])


@router.websocket("/ws/realtime")
async def realtime_ws(websocket: WebSocket) -> None:
    """Accept base64 image frames and return fused realtime inference results."""
    await websocket.accept()
    pipeline = inference_service.create_realtime_session()
    try:
        await websocket.send_json(
            {
                "status": "ready",
                "message": "Realtime inference websocket connected. Send JSON with base64 'image'.",
            }
        )

        while True:
            payload = await websocket.receive_json()
            try:
                response = inference_service.process_realtime_payload(payload, pipeline)
            except Exception as error:
                response = {
                    "status": "error",
                    "message": str(error),
                    "frame_id": payload.get("frame_id"),
                }
            await websocket.send_json(response)
    except WebSocketDisconnect:
        return
    finally:
        inference_service.close_realtime_session(pipeline)
