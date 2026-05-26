"""Service layer for unified rule-based and model-enhanced inference."""

from __future__ import annotations

import base64
import shutil
from datetime import datetime
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import cv2
import numpy as np
from fastapi import UploadFile

from backend.schemas.result import (
    FusionSummary,
    ImageInferenceResponse,
    ModelInferenceSummary,
    RuleInferenceResult,
    UnifiedInferenceResult,
    VideoInferenceResponse,
    VideoInferenceSummary,
)
from inference.common_pipeline import CommonInferencePipeline
from inference.fusion_engine import FusionEngine
from inference.model_runner import ClassificationModelRunner


class InferenceService:
    """Thin orchestration layer above the unified inference stack."""

    def __init__(
        self,
        config_path: str | Path = "configs/mvp.yaml",
        model_checkpoint_path: str | Path = "training/outputs/mobilenetv3_baseline/checkpoints/best.pt",
        model_config_path: str | Path = "training/configs/base.yaml",
    ) -> None:
        """Store config paths, output roots, and initialize reusable model fusion components."""
        self.config_path = Path(config_path)
        self.model_checkpoint_path = Path(model_checkpoint_path)
        self.model_config_path = Path(model_config_path)
        self.output_root = Path("outputs/api_inference")
        self.output_root.mkdir(parents=True, exist_ok=True)

        rule_config = self._load_rule_config()
        self.fusion_engine = FusionEngine(rule_config.get("fusion"))
        self.model_runner = self._create_model_runner()

    async def infer_image(
        self,
        file: UploadFile,
        save_visualization: bool = False,
    ) -> ImageInferenceResponse:
        """Run image inference and return unified fused output."""
        with TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            input_path = temp_root / self._safe_filename(file.filename or "image.jpg")
            await self._save_upload_file(file, input_path)

            frame = cv2.imread(str(input_path))
            if frame is None:
                raise FileNotFoundError(f"Failed to read uploaded image: {input_path}")

            pipeline = CommonInferencePipeline(self._load_rule_config(), static_image_mode=True)
            try:
                frame_result = pipeline.process_frame(frame, frame_id=0, timestamp=0.0, draw_overlay=save_visualization)
            finally:
                pipeline.close()

            fused_result = self._fuse_frame_result(
                frame_result,
                frame=frame,
                allow_full_frame_fallback=True,
                prefer_full_frame_model=True,
                mode="image",
            )
            visualization_path = None
            if save_visualization and frame_result.annotated_frame is not None:
                visualization_path = self._build_output_path("images", input_path.suffix or ".jpg")
                cv2.imwrite(str(visualization_path), frame_result.annotated_frame)

        return ImageInferenceResponse(
            status="success",
            message="image inference completed",
            result=fused_result,
            visualization_path=str(visualization_path) if visualization_path else None,
        )

    async def infer_video(
        self,
        file: UploadFile,
        save_visualization: bool = False,
        include_frames: bool = False,
    ) -> VideoInferenceResponse:
        """Run video inference and return unified fused outputs."""
        with TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            input_path = temp_root / self._safe_filename(file.filename or "video.mp4")
            await self._save_upload_file(file, input_path)

            capture = cv2.VideoCapture(str(input_path))
            if not capture.isOpened():
                raise FileNotFoundError(f"Failed to open uploaded video: {input_path}")

            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

            visualization_path = self._build_output_path("videos", input_path.suffix or ".mp4") if save_visualization else None
            writer = None
            if visualization_path:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(visualization_path), fourcc, fps if fps > 0 else 15.0, (width, height))

            pipeline = CommonInferencePipeline(self._load_rule_config())
            fused_results: list[UnifiedInferenceResult] = []
            events: list[dict[str, Any]] = []
            active_events: dict[str, dict[str, Any]] = {}
            failed_frames = 0

            try:
                frame_id = 0
                while True:
                    success, frame = capture.read()
                    if not success:
                        break

                    timestamp_seconds = float(frame_id / fps) if fps > 0 else float(frame_id)
                    try:
                        frame_result = pipeline.process_frame(
                            frame,
                            frame_id=frame_id,
                            timestamp=timestamp_seconds,
                            draw_overlay=bool(writer),
                        )
                        fused = self._fuse_frame_result(
                            frame_result,
                            frame=frame,
                            allow_full_frame_fallback=True,
                            prefer_full_frame_model=True,
                            mode="video",
                        )
                        fused_results.append(fused)
                        self._update_events(active_events, events, fused, frame_id, timestamp_seconds)

                        if writer is not None:
                            annotated = frame_result.annotated_frame if frame_result.annotated_frame is not None else frame
                            writer.write(annotated)
                    except Exception:
                        failed_frames += 1
                    frame_id += 1
            finally:
                capture.release()
                pipeline.close()
                if writer is not None:
                    writer.release()

        self._flush_events(active_events, events, fused_results)
        average_risk = (
            sum(max(result.fatigue_score, result.distraction_score) for result in fused_results) / len(fused_results)
            if fused_results
            else 0.0
        )
        peak_result = max(
            fused_results,
            key=lambda item: max(item.fatigue_score, item.distraction_score),
            default=None,
        )
        summary = VideoInferenceSummary(
            total_frames=total_frames,
            processed_frames=len(fused_results),
            failed_frames=failed_frames,
            fps=fps,
            average_risk_score=round(float(average_risk), 3),
            peak_risk_score=round(float(max(peak_result.fatigue_score, peak_result.distraction_score)), 3) if peak_result else 0.0,
            peak_risk_level=peak_result.risk_level if peak_result else "normal",
            event_count=len(events),
            saved_visualization=bool(visualization_path),
        )

        return VideoInferenceResponse(
            status="success" if failed_frames == 0 else "partial_success",
            message="video inference completed" if failed_frames == 0 else "video inference completed with frame-level failures",
            summary=summary,
            results=fused_results if include_frames else [],
            events=events,
            visualization_path=str(visualization_path) if visualization_path else None,
        )

    def create_realtime_session(self) -> CommonInferencePipeline:
        """Create a persistent realtime pipeline for one websocket session."""
        return CommonInferencePipeline(self._load_rule_config())

    def close_realtime_session(self, pipeline: CommonInferencePipeline) -> None:
        """Close a realtime pipeline session."""
        pipeline.close()

    def process_realtime_payload(
        self,
        payload: dict[str, Any],
        pipeline: CommonInferencePipeline,
    ) -> dict[str, Any]:
        """Process one websocket payload and return fused realtime output."""
        frame = self._decode_frame_payload(payload)
        frame_id = int(payload.get("frame_id", 0))
        timestamp = payload.get("timestamp")
        frame_result = pipeline.process_frame(
            frame,
            frame_id=frame_id,
            timestamp=timestamp,
            draw_overlay=False,
        )
        fused_result = self._fuse_frame_result(frame_result, frame=frame, mode="realtime")
        return {
            "status": "success",
            "frame_id": frame_id,
            "result": fused_result.model_dump(),
        }

    async def _save_upload_file(self, file: UploadFile, destination: Path) -> None:
        """Persist an uploaded file to a local path."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        await file.close()

    def _safe_filename(self, filename: str) -> str:
        """Reduce an uploaded filename to a safe basename."""
        return Path(filename).name

    def _decode_frame_payload(self, payload: dict[str, Any]) -> np.ndarray:
        """Decode base64 websocket frame payload into an OpenCV BGR image."""
        image_data = payload.get("image")
        if not isinstance(image_data, str) or not image_data:
            raise ValueError("WebSocket payload must include a non-empty 'image' field.")

        encoded = image_data.split(",", 1)[1] if "," in image_data else image_data
        raw_bytes = base64.b64decode(encoded)
        array = np.frombuffer(raw_bytes, dtype=np.uint8)
        frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Failed to decode frame bytes into an image.")
        return frame

    def _build_output_path(self, subdir: str, suffix: str) -> Path:
        """Build a timestamped output path for saved visualizations."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_dir = self.output_root / subdir
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir / f"{timestamp}{suffix}"

    def _create_model_runner(self) -> ClassificationModelRunner | None:
        """Create the optional classifier runner if the checkpoint is available."""
        if not self.model_checkpoint_path.exists():
            return None
        try:
            return ClassificationModelRunner(
                checkpoint_path=self.model_checkpoint_path,
                config_path=self.model_config_path,
                device="auto",
            )
        except Exception:
            return None

    def _load_rule_config(self) -> dict[str, Any]:
        """Load the rule-based config once per request pipeline."""
        import yaml

        with self.config_path.open("r", encoding="utf-8") as config_file:
            return yaml.safe_load(config_file)

    def _build_rule_payload(self, frame_result: Any) -> dict[str, Any]:
        """Convert a frame inference result into the fusion engine's rule payload."""
        status_labels = list(frame_result.status_labels)
        reasons = list(frame_result.reasons)
        alerts: list[str] = ["alarm_on"] if bool(frame_result.alarm_on) else []
        alerts.extend(label for label in status_labels if label not in {"normal", "no_face"})

        return {
            "fatigue_score": float(frame_result.fatigue_score),
            "distraction_score": float(frame_result.distraction_score),
            "risk_score": float(frame_result.risk_score),
            "risk_level": str(frame_result.risk_level),
            "status_labels": status_labels,
            "reasons": reasons,
            "alarm_on": bool(frame_result.alarm_on),
            "signals": {
                "face_detected": bool(frame_result.face_detected),
                "ear": float(frame_result.ear),
                "mar": float(frame_result.mar),
                "yaw": float(frame_result.yaw),
                "pitch": float(frame_result.pitch),
                "roll": float(frame_result.roll),
                "eye_closed_rule": "eye_closed" in status_labels,
                "yawn_rule": "yawning" in status_labels,
                "head_turned_rule": "head_turned" in status_labels,
                "head_down_rule": "head_down" in status_labels,
                "attention_shift_rule": "head_turned" in status_labels or "head_down" in status_labels,
                "status_labels": status_labels,
                "reasons": reasons,
            },
            "alerts": list(dict.fromkeys(alerts)),
        }

    def _fuse_frame_result(
        self,
        frame_result: Any,
        frame: Any,
        allow_full_frame_fallback: bool = False,
        prefer_full_frame_model: bool = False,
        mode: str = "realtime",
    ) -> UnifiedInferenceResult:
        """Run optional model inference and fuse with rule signals."""
        model_probs = {}
        model_summary = ModelInferenceSummary(enabled=self.model_runner is not None)
        if self.model_runner is not None:
            used_face_roi = not prefer_full_frame_model
            model_frame = frame if prefer_full_frame_model else self._extract_model_roi(frame, frame_result)
            if model_frame is None and allow_full_frame_fallback:
                model_frame = frame
                used_face_roi = False
            if model_frame is not None:
                model_output = self.model_runner.predict_from_bgr_frame(model_frame)
                model_probs = model_output.probabilities
                best_label, best_probability = max(
                    model_probs.items(),
                    key=lambda item: item[1],
                    default=(None, 0.0),
                )
                model_summary = ModelInferenceSummary(
                    enabled=True,
                    face_roi_used=used_face_roi,
                    predicted_label=best_label,
                    predicted_confidence=float(best_probability),
                )

        rule_payload = self._build_rule_payload(frame_result)
        fused = self.fusion_engine.fuse(
            rule_result=rule_payload,
            model_probs=model_probs,
            timestamp=str(frame_result.timestamp) if frame_result.timestamp is not None else None,
            mode=mode,
        )
        rule_summary = RuleInferenceResult(
            fatigue_score=float(rule_payload["fatigue_score"]),
            distraction_score=float(rule_payload["distraction_score"]),
            risk_score=float(rule_payload["risk_score"]),
            risk_level=str(rule_payload["risk_level"]),  # type: ignore[arg-type]
            status_labels=[str(item) for item in list(rule_payload["status_labels"])],
            reasons=[str(item) for item in list(rule_payload["reasons"])],
            alarm_on=bool(rule_payload["alarm_on"]),
        )
        fusion_summary = FusionSummary(
            fatigue_score=float(fused["fatigue_score"]),
            distraction_score=float(fused["distraction_score"]),
            risk_level=str(fused["risk_level"]),  # type: ignore[arg-type]
            alerts=[str(item) for item in list(fused.get("alerts", []))],
        )

        return UnifiedInferenceResult(
            fatigue_score=float(fused["fatigue_score"]),
            distraction_score=float(fused["distraction_score"]),
            risk_level=str(fused["risk_level"]),  # type: ignore[arg-type]
            signals=dict(fused.get("signals", {})),
            model_probs={key: float(value) for key, value in dict(fused.get("model_probs", {})).items()},
            alerts=[str(item) for item in list(fused.get("alerts", []))],
            timestamp=str(fused["timestamp"]),
            rule_result=rule_summary,
            model_result=model_summary,
            fusion_result=fusion_summary,
        )

    def _extract_model_roi(self, frame: Any, frame_result: Any) -> Any | None:
        """Prefer a face ROI for classifier inference and skip when no valid face exists."""
        if not bool(getattr(frame_result, "face_detected", False)):
            return None

        bbox = getattr(frame_result, "face_bbox", None)
        if not bbox:
            return None

        x1, y1, x2, y2 = [int(value) for value in bbox]
        frame_height, frame_width = frame.shape[:2]
        x1 = max(0, min(frame_width - 1, x1))
        y1 = max(0, min(frame_height - 1, y1))
        x2 = max(0, min(frame_width, x2))
        y2 = max(0, min(frame_height, y2))

        if x2 - x1 < 32 or y2 - y1 < 32:
            return None

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None
        return roi

    def _update_events(
        self,
        active_events: dict[str, dict[str, Any]],
        completed_events: list[dict[str, Any]],
        result: UnifiedInferenceResult,
        frame_id: int,
        timestamp_seconds: float,
    ) -> None:
        """Update coarse event segments from fused alerts."""
        event_labels = {alert for alert in result.alerts if alert not in {"alarm_on"}}

        for label in list(active_events.keys()):
            if label not in event_labels:
                event = active_events.pop(label)
                event["end_frame"] = frame_id
                event["end_time"] = timestamp_seconds
                event["duration_seconds"] = round(max(timestamp_seconds - float(event["start_time"]), 0.0), 3)
                completed_events.append(event)

        for label in event_labels:
            if label not in active_events:
                active_events[label] = {
                    "event_type": label,
                    "start_frame": frame_id,
                    "end_frame": frame_id,
                    "start_time": timestamp_seconds,
                    "end_time": timestamp_seconds,
                    "peak_risk_level": result.risk_level,
                    "peak_fatigue_score": result.fatigue_score,
                    "peak_distraction_score": result.distraction_score,
                }
            else:
                active = active_events[label]
                active["end_frame"] = frame_id
                active["end_time"] = timestamp_seconds
                if max(result.fatigue_score, result.distraction_score) >= max(
                    float(active["peak_fatigue_score"]),
                    float(active["peak_distraction_score"]),
                ):
                    active["peak_risk_level"] = result.risk_level
                    active["peak_fatigue_score"] = result.fatigue_score
                    active["peak_distraction_score"] = result.distraction_score

    def _flush_events(
        self,
        active_events: dict[str, dict[str, Any]],
        completed_events: list[dict[str, Any]],
        fused_results: list[UnifiedInferenceResult],
    ) -> None:
        """Flush remaining open events at the end of a video."""
        if not fused_results:
            return
        last_index = len(fused_results) - 1
        last_time = float(last_index)
        for event in active_events.values():
            event["end_frame"] = last_index
            event["end_time"] = last_time
            event["duration_seconds"] = round(max(last_time - float(event["start_time"]), 0.0), 3)
            completed_events.append(event)
        active_events.clear()


inference_service = InferenceService()
