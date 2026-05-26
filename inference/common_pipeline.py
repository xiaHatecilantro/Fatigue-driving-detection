"""Shared single-frame inference pipeline for realtime, image, and video modes."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import cv2
import yaml

from cv.features.eye_features import compute_average_ear
from cv.features.head_pose import HeadPoseResult, estimate_head_pose
from cv.features.mouth_features import compute_mar
from cv.scoring.risk_rules import RiskResult, RiskSignals, RuleBasedRiskScorer, TemporalState

try:
    import mediapipe as mp
except ImportError:  # pragma: no cover - runtime dependency
    mp = None


Point2D = tuple[float, float]


@dataclass(slots=True)
class FrameInferenceResult:
    """Structured result for a single processed frame."""

    frame_id: int | None
    timestamp: float | None
    face_detected: bool
    face_bbox: tuple[int, int, int, int] | None
    landmarks: list[Point2D]
    ear: float
    mar: float
    yaw: float
    pitch: float
    roll: float
    fatigue_score: float
    distraction_score: float
    risk_score: float
    risk_level: str
    status_labels: list[str]
    reasons: list[str]
    alarm_on: bool
    annotated_frame: Any | None = None

    def to_dict(self, include_landmarks: bool = True) -> dict[str, Any]:
        """Convert the result to a JSON-safe dictionary."""
        payload = {
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "face": {
                "detected": self.face_detected,
                "landmarks": self.landmarks if include_landmarks else [],
            },
            "features": {
                "ear": self.ear,
                "mar": self.mar,
                "yaw": self.yaw,
                "pitch": self.pitch,
                "roll": self.roll,
            },
            "scores": {
                "fatigue_score": self.fatigue_score,
                "distraction_score": self.distraction_score,
                "risk_score": self.risk_score,
            },
            "result": {
                "risk_level": self.risk_level,
                "status_labels": self.status_labels,
                "reasons": self.reasons,
                "alarm_on": self.alarm_on,
            },
        }
        return payload


class CommonInferencePipeline:
    """Config-driven single-frame detector shared by offline inference workflows."""

    def __init__(self, config: Mapping[str, Any], static_image_mode: bool = False) -> None:
        """Initialize the shared detector pipeline."""
        self.config = config
        self.static_image_mode = static_image_mode
        self.temporal_state = TemporalState()
        self.risk_scorer = RuleBasedRiskScorer(config)
        self.landmark_config = config["landmarks"]
        self.ui_config = config.get("ui", {})
        self.face_mesh = self._create_face_mesh(config.get("mediapipe", {}))

    def _create_face_mesh(self, mediapipe_config: Mapping[str, Any]) -> Any:
        """Create MediaPipe Face Mesh or return None when unavailable."""
        if mp is None:
            return None
        return mp.solutions.face_mesh.FaceMesh(
            static_image_mode=self.static_image_mode,
            max_num_faces=int(mediapipe_config.get("max_num_faces", 1)),
            refine_landmarks=bool(mediapipe_config.get("refine_landmarks", False)),
            min_detection_confidence=float(mediapipe_config.get("min_detection_confidence", 0.5)),
            min_tracking_confidence=float(mediapipe_config.get("min_tracking_confidence", 0.5)),
        )

    def close(self) -> None:
        """Release MediaPipe resources."""
        if self.face_mesh is not None:
            self.face_mesh.close()

    def process_frame(
        self,
        frame: Any,
        frame_id: int | None = None,
        timestamp: float | None = None,
        draw_overlay: bool = True,
    ) -> FrameInferenceResult:
        """Run full rule-based inference on a single frame with tolerance for failure."""
        landmarks = self._extract_landmarks(frame)
        if not landmarks:
            self.temporal_state.missing_face_frames += 1
            self.temporal_state.eye_closed_frames = 0
            self.temporal_state.yawn_frames = 0
            self.temporal_state.head_turn_frames = 0
            self.temporal_state.head_down_frames = 0

            signals = RiskSignals(
                face_detected=False,
                ear=0.0,
                mar=0.0,
                yaw=0.0,
                pitch=0.0,
                roll=0.0,
                eye_closed=False,
                yawning=False,
                head_turned=False,
                head_down=False,
            )
            risk = self.risk_scorer.score(signals, self.temporal_state)
            result = FrameInferenceResult(
                frame_id=frame_id,
                timestamp=timestamp,
                face_detected=False,
                face_bbox=None,
                landmarks=[],
                ear=0.0,
                mar=0.0,
                yaw=0.0,
                pitch=0.0,
                roll=0.0,
                fatigue_score=risk.fatigue_score,
                distraction_score=risk.distraction_score,
                risk_score=risk.risk_score,
                risk_level=risk.risk_level,
                status_labels=risk.status_labels,
                reasons=risk.reasons,
                alarm_on=risk.alarm_on,
            )
            result.annotated_frame = self._draw_overlay(frame, result) if draw_overlay else None
            return result

        self.temporal_state.missing_face_frames = 0
        left_ear, right_ear, ear = compute_average_ear(
            landmarks=landmarks,
            left_eye_indices=self.landmark_config["left_eye"],
            right_eye_indices=self.landmark_config["right_eye"],
        )
        mar = compute_mar(landmarks, self.landmark_config["mouth"])
        head_pose = estimate_head_pose(
            landmarks=landmarks,
            pose_config=self.config.get("head_pose", {}),
            landmark_indices=self.landmark_config["pose_points"],
        )

        thresholds = self.config.get("thresholds", {})
        eye_closed = ear > 0.0 and ear <= float(thresholds.get("ear_closed", 0.22))
        yawning = mar >= float(thresholds.get("mar_yawn", 0.60))

        self.temporal_state.eye_closed_frames = self._advance_counter(self.temporal_state.eye_closed_frames, eye_closed)
        self.temporal_state.yawn_frames = self._advance_counter(self.temporal_state.yawn_frames, yawning)
        self.temporal_state.head_turn_frames = self._advance_counter(self.temporal_state.head_turn_frames, head_pose.head_turned)
        self.temporal_state.head_down_frames = self._advance_counter(self.temporal_state.head_down_frames, head_pose.head_down)

        signals = RiskSignals(
            face_detected=True,
            ear=ear,
            mar=mar,
            yaw=head_pose.yaw,
            pitch=head_pose.pitch,
            roll=head_pose.roll,
            eye_closed=eye_closed,
            yawning=yawning,
            head_turned=head_pose.head_turned,
            head_down=head_pose.head_down,
        )
        risk = self.risk_scorer.score(signals, self.temporal_state)
        result = FrameInferenceResult(
            frame_id=frame_id,
            timestamp=timestamp,
            face_detected=True,
            face_bbox=self._compute_face_bbox(frame.shape[:2], landmarks),
            landmarks=landmarks,
            ear=ear,
            mar=mar,
            yaw=head_pose.yaw,
            pitch=head_pose.pitch,
            roll=head_pose.roll,
            fatigue_score=risk.fatigue_score,
            distraction_score=risk.distraction_score,
            risk_score=risk.risk_score,
            risk_level=risk.risk_level,
            status_labels=risk.status_labels,
            reasons=risk.reasons,
            alarm_on=risk.alarm_on,
        )
        result.annotated_frame = self._draw_overlay(
            frame,
            result,
            left_ear=left_ear,
            right_ear=right_ear,
            head_pose=head_pose,
        ) if draw_overlay else None
        return result

    def _compute_face_bbox(
        self,
        frame_shape: tuple[int, int],
        landmarks: list[Point2D],
    ) -> tuple[int, int, int, int] | None:
        """Compute a clamped face bounding box from landmarks."""
        if not landmarks:
            return None

        frame_height, frame_width = frame_shape
        x_coords = [point[0] for point in landmarks]
        y_coords = [point[1] for point in landmarks]
        min_x = min(x_coords)
        max_x = max(x_coords)
        min_y = min(y_coords)
        max_y = max(y_coords)

        width = max_x - min_x
        height = max_y - min_y
        if width <= 1 or height <= 1:
            return None

        # Add a modest margin so the classifier sees face context rather than only landmarks.
        expand_x = width * 0.18
        expand_y = height * 0.22

        x1 = max(0, int(min_x - expand_x))
        y1 = max(0, int(min_y - expand_y))
        x2 = min(frame_width - 1, int(max_x + expand_x))
        y2 = min(frame_height - 1, int(max_y + expand_y))

        if x2 <= x1 or y2 <= y1:
            return None
        return (x1, y1, x2, y2)

    def _extract_landmarks(self, frame: Any) -> list[Point2D]:
        """Extract image-space landmarks from a BGR frame."""
        if self.face_mesh is None:
            return []

        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.face_mesh.process(rgb_frame)
        except Exception:
            return []

        if not result.multi_face_landmarks:
            return []

        frame_height, frame_width = frame.shape[:2]
        face_landmarks = result.multi_face_landmarks[0].landmark
        return [
            (float(landmark.x * frame_width), float(landmark.y * frame_height))
            for landmark in face_landmarks
        ]

    def _advance_counter(self, current_value: int, activated: bool) -> int:
        """Advance a temporal counter while a condition is active."""
        return current_value + 1 if activated else 0

    def _draw_overlay(
        self,
        frame: Any,
        result: FrameInferenceResult,
        left_ear: float = 0.0,
        right_ear: float = 0.0,
        head_pose: HeadPoseResult | None = None,
    ) -> Any:
        """Render landmarks, scores, and alarm text onto a copy of the frame."""
        output = frame.copy()

        if self.ui_config.get("show_landmarks", True):
            for x_coord, y_coord in result.landmarks:
                cv2.circle(output, (int(x_coord), int(y_coord)), 1, (0, 255, 0), -1)

        pose = head_pose or HeadPoseResult(result.yaw, result.pitch, result.roll, False, False)
        lines = [
            f"EAR: {result.ear:.3f}  L:{left_ear:.3f} R:{right_ear:.3f}",
            f"MAR: {result.mar:.3f}",
            f"Yaw/Pitch/Roll: {pose.yaw:.1f}/{pose.pitch:.1f}/{pose.roll:.1f}",
            f"Fatigue: {result.fatigue_score:.1f}",
            f"Distraction: {result.distraction_score:.1f}",
            f"Risk: {result.risk_score:.1f} ({result.risk_level})",
            f"Status: {', '.join(result.status_labels)}",
        ]
        if not result.face_detected:
            lines.insert(0, "Face: not detected")

        text_color = (0, 255, 0)
        if result.risk_level == "mild":
            text_color = (0, 255, 255)
        elif result.risk_level == "moderate":
            text_color = (0, 165, 255)
        elif result.risk_level == "severe":
            text_color = (0, 0, 255)

        for index, line in enumerate(lines):
            y_coord = 25 + index * 24
            cv2.putText(output, line, (12, y_coord), cv2.FONT_HERSHEY_SIMPLEX, 0.65, text_color, 2, cv2.LINE_AA)

        if result.alarm_on and self.ui_config.get("show_alarm", True):
            cv2.rectangle(output, (0, 0), (output.shape[1] - 1, output.shape[0] - 1), (0, 0, 255), 3)
            cv2.putText(
                output,
                f"ALARM: {result.risk_level.upper()}",
                (12, output.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                3,
                cv2.LINE_AA,
            )
        return output


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file."""
    with Path(config_path).open("r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def ensure_parent_dir(output_path: str | Path) -> None:
    """Create the parent directory for an output file if needed."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)


def write_json_file(output_path: str | Path, payload: Mapping[str, Any]) -> None:
    """Persist JSON payload to disk with UTF-8 encoding."""
    ensure_parent_dir(output_path)
    with Path(output_path).open("w", encoding="utf-8") as json_file:
        json.dump(payload, json_file, ensure_ascii=False, indent=2)
