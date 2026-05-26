"""Mouth-related geometric features for yawning detection."""

from __future__ import annotations

from math import dist
from typing import Sequence


Point2D = tuple[float, float]


def _safe_distance(point_a: Point2D, point_b: Point2D) -> float:
    """Return Euclidean distance between two points."""
    return float(dist(point_a, point_b))


def compute_mouth_aspect_ratio(mouth_points: Sequence[Point2D]) -> float:
    """Compute MAR from four ordered mouth points.

    The expected point order is: left_corner, upper_lip, lower_lip, right_corner.
    """
    if len(mouth_points) != 4:
        return 0.0

    width = _safe_distance(mouth_points[0], mouth_points[3])
    if width <= 1e-6:
        return 0.0

    height = _safe_distance(mouth_points[1], mouth_points[2])
    return float(height / width)


def compute_mar(landmarks: Sequence[Point2D], mouth_indices: Sequence[int]) -> float:
    """Compute MAR from a landmark set and configured mouth indices."""
    try:
        mouth_points = [landmarks[index] for index in mouth_indices]
    except (IndexError, TypeError):
        return 0.0
    return compute_mouth_aspect_ratio(mouth_points)
