"""Minimal runnable realtime detector for rule-based fatigue and distraction MVP."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import yaml

from cv.features.eye_features import compute_average_ear
from cv.features.head_pose import HeadPoseResult, estimate_head_pose
from cv.features.mouth_features import compute_mar
from cv.scoring.risk_rules import RiskResult, RiskSignals, RuleBasedRiskScorer, TemporalState

try:
    import mediapipe as mp
except ImportError:  # pragma: no cover - depends on runtime environment
    mp = None


Point2D = tuple[float, float]


@dataclass(slots=True)
class DetectionFrameResult:
    """Frame-level detection output for rendering and debugging."""

    face_detected: bool
    landmarks: list[Point2D]
    ear: float
    mar: float
    head_pose: HeadPoseResult
    risk: RiskResult


class RealtimeDetector:
    """Rule-based realtime detector using OpenCV and MediaPipe Face Mesh."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        """Initialize detector runtime with config-driven thresholds and indices."""
        self.config = config
        self.temporal_state = TemporalState()
        self.risk_scorer = RuleBasedRiskScorer(config)
        self.landmark_config = config["landmarks"]
        self.ui_config = config.get("ui", {})
        self.face_mesh = self._create_face_mesh(config.get("mediapipe", {}))

    def _create_face_mesh(self, mediapipe_config: Mapping[str, Any]) -> Any:
        """Create MediaPipe Face Mesh if the dependency is available."""
        if mp is None:
            return None
        return mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=int(mediapipe_config.get("max_num_faces", 1)),
            refine_landmarks=bool(mediapipe_config.get("refine_landmarks", False)),
            min_detection_confidence=float(mediapipe_config.get("min_detection_confidence", 0.5)),
            min_tracking_confidence=float(mediapipe_config.get("min_tracking_confidence", 0.5)),
        )

    def close(self) -> None:
        """Release detector-side resources."""
        if self.face_mesh is not None:
            self.face_mesh.close()

    def process_frame(self, frame: Any) -> tuple[Any, DetectionFrameResult]:
        """Process a single frame and return an annotated frame plus structured result."""
        landmarks = self._extract_landmarks(frame)
        if not landmarks:
            self.temporal_state.missing_face_frames += 1
            self.temporal_state.eye_closed_frames = 0
            self.temporal_state.yawn_frames = 0
            self.temporal_state.head_turn_frames = 0
            self.temporal_state.head_down_frames = 0
            head_pose = HeadPoseResult(0.0, 0.0, 0.0, False, False)
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
            result = DetectionFrameResult(False, [], 0.0, 0.0, head_pose, risk)
            return self._draw_overlay(frame, result), result

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
        result = DetectionFrameResult(True, landmarks, ear, mar, head_pose, risk)
        return self._draw_overlay(frame, result, left_ear=left_ear, right_ear=right_ear), result

    def _extract_landmarks(self, frame: Any) -> list[Point2D]:
        """Extract 2D face landmarks in image coordinates, or return an empty list on failure."""
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
        """Advance a temporal counter while a rule remains active."""
        return current_value + 1 if activated else 0

    def _draw_overlay(
        self,
        frame: Any,
        result: DetectionFrameResult,
        left_ear: float = 0.0,
        right_ear: float = 0.0,
    ) -> Any:
        """Draw landmarks, scores, and alarm text onto the frame."""
        output = frame.copy()

        if self.ui_config.get("show_landmarks", True):
            for x_coord, y_coord in result.landmarks:
                cv2.circle(output, (int(x_coord), int(y_coord)), 1, (0, 255, 0), -1)

        lines = [
            f"EAR: {result.ear:.3f}  L:{left_ear:.3f} R:{right_ear:.3f}",
            f"MAR: {result.mar:.3f}",
            f"Yaw/Pitch/Roll: {result.head_pose.yaw:.1f}/{result.head_pose.pitch:.1f}/{result.head_pose.roll:.1f}",
            f"Fatigue: {result.risk.fatigue_score:.1f}",
            f"Distraction: {result.risk.distraction_score:.1f}",
            f"Risk: {result.risk.risk_score:.1f} ({result.risk.risk_level})",
            f"Status: {', '.join(result.risk.status_labels)}",
        ]
        if not result.face_detected:
            lines.insert(0, "Face: not detected")

        text_color = (0, 255, 0)
        if result.risk.risk_level == "mild":
            text_color = (0, 255, 255)
        elif result.risk.risk_level == "moderate":
            text_color = (0, 165, 255)
        elif result.risk.risk_level == "severe":
            text_color = (0, 0, 255)

        for index, line in enumerate(lines):
            y_coord = 25 + index * 24
            cv2.putText(output, line, (12, y_coord), cv2.FONT_HERSHEY_SIMPLEX, 0.65, text_color, 2, cv2.LINE_AA)

        if result.risk.alarm_on and self.ui_config.get("show_alarm", True):
            cv2.rectangle(output, (0, 0), (output.shape[1] - 1, output.shape[0] - 1), (0, 0, 255), 3)
            cv2.putText(
                output,
                f"ALARM: {result.risk.risk_level.upper()}",
                (12, output.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                3,
                cv2.LINE_AA,
            )

        return output


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load detector configuration from a YAML file."""
    with Path(config_path).open("r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def run_camera_loop(config: Mapping[str, Any]) -> None:
    """Start the realtime camera loop and render rule-based detection results."""
    camera_config = config.get("camera", {})
    detector = RealtimeDetector(config)
    cap = cv2.VideoCapture(int(camera_config.get("index", 0)))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(camera_config.get("width", 640)))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(camera_config.get("height", 480)))

    if not cap.isOpened():
        detector.close()
        raise RuntimeError("Failed to open camera. Check camera index or permissions.")

    window_name = str(config.get("window_name", "Driver Fatigue & Distraction MVP"))
    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            annotated_frame, _ = detector.process_frame(frame)
            cv2.imshow(window_name, annotated_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        cap.release()
        detector.close()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the realtime detector."""
    parser = argparse.ArgumentParser(description="Run the MVP realtime fatigue detector.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/mvp.yaml",
        help="Path to the YAML config file.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    config = load_config(args.config)
    run_camera_loop(config)


if __name__ == "__main__":
    main()
