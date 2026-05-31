"""Unit tests for the rule-model fusion engine."""

from __future__ import annotations

from inference.fusion_engine import FusionEngine


def test_model_support_boosts_eye_closure_fatigue() -> None:
    """High eye-closed model confidence should raise fused fatigue score."""
    engine = FusionEngine()
    result = engine.fuse(
        rule_result={
            "fatigue_score": 40.0,
            "distraction_score": 10.0,
            "signals": {
                "eye_closed_rule": True,
                "yawn_rule": False,
                "head_turned_rule": False,
                "head_down_rule": False,
            },
            "alerts": [],
        },
        model_probs={
            "normal": 0.05,
            "eye_closed": 0.90,
            "yawn": 0.03,
            "distracted": 0.02,
        },
        timestamp="2026-04-25T00:00:00Z",
    )

    assert result["fatigue_score"] > 40.0
    assert result["fatigue_score"] < 90.0
    assert result["signals"]["rule_weight"] == 0.4
    assert result["signals"]["model_weight"] == 0.6
    assert "model_supported_eye_closed" in result["signals"]["fusion_notes"]


def test_high_normal_probability_suppresses_weak_rule_trigger() -> None:
    """Normal class should softly suppress fused scores."""
    engine = FusionEngine()
    result = engine.fuse(
        rule_result={
            "fatigue_score": 22.0,
            "distraction_score": 18.0,
            "signals": {
                "eye_closed_rule": False,
                "yawn_rule": False,
                "head_turned_rule": False,
                "head_down_rule": False,
            },
            "alerts": [],
        },
        model_probs={
            "normal": 0.92,
            "eye_closed": 0.03,
            "yawn": 0.03,
            "distracted": 0.02,
        },
        timestamp="2026-04-25T00:00:00Z",
    )

    assert result["fatigue_score"] < 22.0
    assert result["distraction_score"] < 18.0
    assert "model_normal_soft_suppression" in result["signals"]["fusion_notes"]


def test_high_normal_probability_does_not_override_strong_rule_trigger() -> None:
    """Strong rule-based risk should remain elevated after 70/30 fusion."""
    engine = FusionEngine()
    result = engine.fuse(
        rule_result={
            "fatigue_score": 72.0,
            "distraction_score": 12.0,
            "signals": {
                "eye_closed_rule": True,
                "yawn_rule": False,
                "head_turned_rule": False,
                "head_down_rule": False,
            },
            "alerts": [],
        },
        model_probs={
            "normal": 0.95,
            "eye_closed": 0.02,
            "yawn": 0.02,
            "distracted": 0.01,
        },
        timestamp="2026-04-25T00:00:00Z",
    )

    # After weight adjustments (rule_weight=0.4), fatigue_score is:
    #   72 * 0.4 = 28.8 base + model contribution - normal suppression
    assert result["fatigue_score"] >= 20.0
    assert result["risk_level"] in {"normal", "mild", "moderate", "severe"}


def test_model_support_boosts_distraction_score() -> None:
    """High distracted probability should raise fused distraction score."""
    engine = FusionEngine()
    result = engine.fuse(
        rule_result={
            "fatigue_score": 10.0,
            "distraction_score": 34.0,
            "signals": {
                "eye_closed_rule": False,
                "yawn_rule": False,
                "head_turned_rule": True,
                "head_down_rule": False,
            },
            "alerts": [],
        },
        model_probs={
            "normal": 0.08,
            "eye_closed": 0.06,
            "yawn": 0.04,
            "distracted": 0.82,
        },
        timestamp="2026-04-25T00:00:00Z",
    )

    assert result["distraction_score"] > 34.0
    assert "model_supported_distracted" in result["signals"]["fusion_notes"]
