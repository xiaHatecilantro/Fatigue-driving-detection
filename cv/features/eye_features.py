"""Eye-related geometric features for fatigue detection."""

from __future__ import annotations

from math import dist
from typing import Sequence


Point2D = tuple[float, float]


def _safe_distance(point_a: Point2D, point_b: Point2D) -> float:
    """Return Euclidean distance between two points."""
    return float(dist(point_a, point_b))


def compute_eye_aspect_ratio(eye_points: Sequence[Point2D]) -> float:
    """Compute EAR from six ordered eye contour points.

    The expected point order matches the common EAR definition:
    p1, p2, p3, p4, p5, p6 where p1-p4 is the horizontal line and
    (p2, p6), (p3, p5) are the vertical pairs.
    """
    if len(eye_points) != 6:
        return 0.0

    horizontal = _safe_distance(eye_points[0], eye_points[3])
    if horizontal <= 1e-6:
        return 0.0

    vertical_a = _safe_distance(eye_points[1], eye_points[5])
    vertical_b = _safe_distance(eye_points[2], eye_points[4])
    return float((vertical_a + vertical_b) / (2.0 * horizontal))


def compute_average_ear(
    landmarks: Sequence[Point2D],
    left_eye_indices: Sequence[int],
    right_eye_indices: Sequence[int],
) -> tuple[float, float, float]:
    """Compute left, right, and averaged EAR values from face landmarks."""
    try:
        left_eye = [landmarks[index] for index in left_eye_indices]
        right_eye = [landmarks[index] for index in right_eye_indices]
    except (IndexError, TypeError):
        return 0.0, 0.0, 0.0

    left_ear = compute_eye_aspect_ratio(left_eye)
    right_ear = compute_eye_aspect_ratio(right_eye)
    average_ear = (left_ear + right_ear) / 2.0
    return float(left_ear), float(right_ear), float(average_ear)
