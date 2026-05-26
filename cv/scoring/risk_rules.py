"""Rule-based fusion for fatigue and distraction scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(slots=True)
class RiskSignals:
    """Per-frame signals consumed by the rule engine."""

    face_detected: bool
    ear: float
    mar: float
    yaw: float
    pitch: float
    roll: float
    eye_closed: bool
    yawning: bool
    head_turned: bool
    head_down: bool


@dataclass(slots=True)
class TemporalState:
    """Short-window counters used by the rule engine."""

    eye_closed_frames: int = 0
    yawn_frames: int = 0
    head_turn_frames: int = 0
    head_down_frames: int = 0
    missing_face_frames: int = 0


@dataclass(slots=True)
class RiskResult:
    """Fused business output for display or downstream JSON formatting."""

    fatigue_score: float
    distraction_score: float
    risk_score: float
    risk_level: str
    alarm_on: bool
    status_labels: list[str]
    reasons: list[str]


def _clamp_score(score: float) -> float:
    """Clamp a risk score to the 0-100 range."""
    return max(0.0, min(100.0, float(score)))


def _read_level_thresholds(config: Mapping[str, float | int]) -> tuple[float, float, float]:
    """Read and normalize level thresholds from config."""
    mild = float(config.get("mild", 30.0))
    moderate = float(config.get("moderate", 60.0))
    severe = float(config.get("severe", 80.0))
    return mild, moderate, severe


def _to_level(score: float, thresholds: Mapping[str, float | int]) -> str:
    """Map a normalized score to a risk level label."""
    mild, moderate, severe = _read_level_thresholds(thresholds)
    if score >= severe:
        return "severe"
    if score >= moderate:
        return "moderate"
    if score >= mild:
        return "mild"
    return "normal"


def _increment(base: float, step: float, frames: int, max_boost: float) -> float:
    """Increase a score with a bounded temporal bonus."""
    return float(base + min(step * max(frames, 0), max_boost))


class RuleBasedRiskScorer:
    """Fuse frame-level signals into fatigue and distraction scores."""

    def __init__(self, config: Mapping[str, object]) -> None:
        """Store scorer configuration."""
        self._config = config

    def score(self, signals: RiskSignals, state: TemporalState) -> RiskResult:
        """Score current frame with graceful fallback when face detection fails."""
        risk_config = self._config.get("risk", {})
        temporal_config = self._config.get("temporal", {})
        thresholds = self._config.get("thresholds", {})
        weights = risk_config.get("weights", {})
        level_thresholds = risk_config.get("level_thresholds", {})
        alarm_threshold = float(risk_config.get("alarm_threshold", 60.0))

        status_labels: list[str] = []
        reasons: list[str] = []

        if not signals.face_detected:
            missing_penalty = float(weights.get("missing_face_penalty", 10.0))
            score = _clamp_score(min(missing_penalty + state.missing_face_frames * 2.0, 30.0))
            return RiskResult(
                fatigue_score=0.0,
                distraction_score=score,
                risk_score=score,
                risk_level=_to_level(score, level_thresholds),
                alarm_on=score >= alarm_threshold,
                status_labels=["no_face"],
                reasons=["face_not_detected"],
            )

        fatigue_score = 0.0
        distraction_score = 0.0

        eye_frames_threshold = int(temporal_config.get("eye_closed_frames", 3))
        yawn_frames_threshold = int(temporal_config.get("yawn_frames", 8))
        head_turn_frames_threshold = int(temporal_config.get("head_turn_frames", 8))
        head_down_frames_threshold = int(temporal_config.get("head_down_frames", 8))

        if signals.eye_closed and state.eye_closed_frames >= eye_frames_threshold:
            fatigue_score = _increment(
                base=float(weights.get("eye_closed_base", 35.0)),
                step=float(weights.get("eye_closed_step", 5.0)),
                frames=state.eye_closed_frames - eye_frames_threshold + 1,
                max_boost=float(weights.get("eye_closed_max_boost", 35.0)),
            )
            status_labels.append("eye_closed")
            reasons.append("long_eye_closure")

        if signals.yawning and state.yawn_frames >= yawn_frames_threshold:
            fatigue_score += _increment(
                base=float(weights.get("yawn_base", 20.0)),
                step=float(weights.get("yawn_step", 3.0)),
                frames=state.yawn_frames - yawn_frames_threshold + 1,
                max_boost=float(weights.get("yawn_max_boost", 20.0)),
            )
            status_labels.append("yawning")
            reasons.append("yawn_detected")

        if signals.head_down and state.head_down_frames >= head_down_frames_threshold:
            head_down_score = _increment(
                base=float(weights.get("head_down_base", 18.0)),
                step=float(weights.get("head_down_step", 3.0)),
                frames=state.head_down_frames - head_down_frames_threshold + 1,
                max_boost=float(weights.get("head_down_max_boost", 20.0)),
            )
            fatigue_score += head_down_score * float(weights.get("head_down_fatigue_ratio", 0.5))
            distraction_score += head_down_score
            status_labels.append("head_down")
            reasons.append("head_down_detected")

        if signals.head_turned and state.head_turn_frames >= head_turn_frames_threshold:
            distraction_score += _increment(
                base=float(weights.get("head_turn_base", 30.0)),
                step=float(weights.get("head_turn_step", 4.0)),
                frames=state.head_turn_frames - head_turn_frames_threshold + 1,
                max_boost=float(weights.get("head_turn_max_boost", 25.0)),
            )
            status_labels.append("head_turned")
            reasons.append("attention_shifted")

        ear_penalty_threshold = float(thresholds.get("ear_closed", 0.22))
        mar_penalty_threshold = float(thresholds.get("mar_yawn", 0.60))

        if signals.ear > 0.0 and signals.ear < ear_penalty_threshold and "eye_closed" not in status_labels:
            fatigue_score += float(weights.get("eye_low_penalty", 8.0))
        if signals.mar > mar_penalty_threshold and "yawning" not in status_labels:
            fatigue_score += float(weights.get("mouth_open_penalty", 5.0))

        fatigue_score = _clamp_score(fatigue_score)
        distraction_score = _clamp_score(distraction_score)
        risk_score = _clamp_score(max(fatigue_score, distraction_score) * 0.7 + (fatigue_score + distraction_score) * 0.15)
        risk_level = _to_level(risk_score, level_thresholds)

        return RiskResult(
            fatigue_score=fatigue_score,
            distraction_score=distraction_score,
            risk_score=risk_score,
            risk_level=risk_level,
            alarm_on=risk_score >= alarm_threshold,
            status_labels=status_labels or ["normal"],
            reasons=reasons or ["stable"],
        )
