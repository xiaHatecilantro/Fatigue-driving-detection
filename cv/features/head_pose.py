"""Lightweight head pose approximation using MediaPipe face landmarks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


Point2D = tuple[float, float]


@dataclass(slots=True)
class HeadPoseResult:
    """Approximate head pose and derived status labels."""

    yaw: float
    pitch: float
    roll: float
    head_turned: bool
    head_down: bool


def _get_point(landmarks: Sequence[Point2D], index: int) -> Point2D | None:
    """Safely read a landmark coordinate."""
    if index < 0 or index >= len(landmarks):
        return None
    return landmarks[index]


def estimate_head_pose(
    landmarks: Sequence[Point2D],
    pose_config: Mapping[str, float | int],
    landmark_indices: Mapping[str, int],
) -> HeadPoseResult:
    """Estimate coarse yaw, pitch, and roll from 2D landmarks.

    This is a lightweight approximation suitable for a competition MVP on CPU.
    """
    nose = _get_point(landmarks, int(landmark_indices["nose_tip"]))
    chin = _get_point(landmarks, int(landmark_indices["chin"]))
    forehead = _get_point(landmarks, int(landmark_indices["forehead"]))
    left_face = _get_point(landmarks, int(landmark_indices["left_face"]))
    right_face = _get_point(landmarks, int(landmark_indices["right_face"]))

    if not all((nose, chin, forehead, left_face, right_face)):
        return HeadPoseResult(0.0, 0.0, 0.0, False, False)

    left_face = left_face or (0.0, 0.0)
    right_face = right_face or (0.0, 0.0)
    nose = nose or (0.0, 0.0)
    chin = chin or (0.0, 0.0)
    forehead = forehead or (0.0, 0.0)

    face_width = max(right_face[0] - left_face[0], 1.0)
    face_height = max(chin[1] - forehead[1], 1.0)
    face_center_x = (left_face[0] + right_face[0]) / 2.0

    yaw_ratio = (nose[0] - face_center_x) / (face_width / 2.0)
    yaw = float(yaw_ratio * float(pose_config.get("yaw_scale", 35.0)))

    pitch_ratio = (nose[1] - forehead[1]) / face_height
    pitch_center = float(pose_config.get("pitch_center_ratio", 0.38))
    pitch = float((pitch_ratio - pitch_center) * float(pose_config.get("pitch_scale", 80.0)))

    dx = chin[0] - forehead[0]
    dy = max(chin[1] - forehead[1], 1.0)
    roll = float((dx / dy) * float(pose_config.get("roll_scale", 45.0)))

    yaw_threshold = float(pose_config.get("yaw_turn_threshold", 18.0))
    pitch_threshold = float(pose_config.get("pitch_down_threshold", 10.0))

    head_turned = abs(yaw) >= yaw_threshold
    head_down = pitch >= pitch_threshold

    return HeadPoseResult(
        yaw=yaw,
        pitch=pitch,
        roll=roll,
        head_turned=head_turned,
        head_down=head_down,
    )
