"""Fusion engine combining rule-based signals with classifier probabilities."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Mapping


DEFAULT_FUSION_CONFIG: dict[str, Any] = {
    "weights": {
        "rule_weight": 0.4,
        "model_weight": 0.6,
    },
    "image_weights": {
        "rule_weight": 0.2,
        "model_weight": 0.8,
    },
    "model_thresholds": {
        "eye_closed_support": 0.65,
        "yawn_support": 0.60,
        "distracted_support": 0.70,
        "normal_override_threshold": 0.80,
    },
    "model_score_mapping": {
        "eye_closed_multiplier": 100.0,
        "yawn_multiplier": 85.0,
        "distracted_multiplier": 100.0,
    },
    "risk_thresholds": {
        "mild": 30.0,
        "moderate": 60.0,
        "severe": 80.0,
    },
    "alerts": {
        "fatigue_warning_threshold": 60.0,
        "distraction_warning_threshold": 60.0,
    }
}


def _deep_merge(base: dict[str, Any], overrides: Mapping[str, Any] | None) -> dict[str, Any]:
    """Deep merge nested dictionaries."""
    if overrides is None:
        return deepcopy(base)

    merged = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _clamp_score(value: float) -> float:
    """Clamp score into the 0-100 range."""
    return max(0.0, min(100.0, float(value)))


def _to_risk_level(score: float, thresholds: Mapping[str, Any]) -> str:
    """Map score to discrete risk level."""
    if score >= float(thresholds.get("severe", 80.0)):
        return "severe"
    if score >= float(thresholds.get("moderate", 60.0)):
        return "moderate"
    if score >= float(thresholds.get("mild", 30.0)):
        return "mild"
    return "normal"


class FusionEngine:
    """Fuse rule-based outputs and classifier probabilities into one service response."""

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        """Store fusion config with sane defaults."""
        self.config = _deep_merge(DEFAULT_FUSION_CONFIG, config)

    def fuse(
        self,
        rule_result: Mapping[str, Any],
        model_probs: Mapping[str, float] | None = None,
        timestamp: str | None = None,
        mode: str = "realtime",
    ) -> dict[str, Any]:
        """Fuse rule and model outputs into the unified JSON schema."""
        weights_cfg = self.config["weights"]
        thresholds = self.config["model_thresholds"]
        model_mapping = self.config["model_score_mapping"]
        risk_thresholds = self.config["risk_thresholds"]
        alert_cfg = self.config["alerts"]
        rule_fatigue = float(rule_result.get("fatigue_score", 0.0))
        rule_distraction = float(rule_result.get("distraction_score", 0.0))

        base_signals = deepcopy(dict(rule_result.get("signals", {})))
        fusion_notes = list(base_signals.get("fusion_notes", []))

        if mode == "image":
            image_weights = dict(self.config.get("image_weights", {}))
            rule_weight = float(image_weights.get("rule_weight", 0.2))
            model_weight = float(image_weights.get("model_weight", 0.8))
            fusion_notes.append("image_mode_model_priority")
        else:
            rule_weight = float(weights_cfg.get("rule_weight", 0.7))
            model_weight = float(weights_cfg.get("model_weight", 0.3))

        alerts = list(rule_result.get("alerts", []))
        probabilities = {
            "normal": 0.0,
            "eye_closed": 0.0,
            "yawn": 0.0,
            "distracted": 0.0,
        }
        if model_probs is not None:
            for key in probabilities:
                probabilities[key] = float(model_probs.get(key, 0.0))

        eye_closed_rule = bool(base_signals.get("eye_closed_rule", False))
        yawn_rule = bool(base_signals.get("yawn_rule", False))
        head_turned_rule = bool(base_signals.get("head_turned_rule", False))
        head_down_rule = bool(base_signals.get("head_down_rule", False))
        distraction_rule = head_turned_rule or head_down_rule or bool(
            base_signals.get("attention_shift_rule", False)
        )

        model_fatigue_score = max(
            probabilities["eye_closed"] * float(model_mapping.get("eye_closed_multiplier", 100.0)),
            probabilities["yawn"] * float(model_mapping.get("yawn_multiplier", 85.0)),
        )
        model_distraction_score = (
            probabilities["distracted"] * float(model_mapping.get("distracted_multiplier", 100.0))
        )

        if eye_closed_rule and probabilities["eye_closed"] >= float(thresholds.get("eye_closed_support", 0.80)):
            fusion_notes.append("model_supported_eye_closed")
        if yawn_rule and probabilities["yawn"] >= float(thresholds.get("yawn_support", 0.75)):
            fusion_notes.append("model_supported_yawn")
        if distraction_rule and probabilities["distracted"] >= float(thresholds.get("distracted_support", 0.75)):
            fusion_notes.append("model_supported_distracted")

        if not eye_closed_rule and probabilities["eye_closed"] >= float(thresholds.get("eye_closed_support", 0.80)):
            fusion_notes.append("model_soft_eye_closed_boost")
        if not yawn_rule and probabilities["yawn"] >= float(thresholds.get("yawn_support", 0.75)):
            fusion_notes.append("model_soft_yawn_boost")
        if not distraction_rule and probabilities["distracted"] >= float(thresholds.get("distracted_support", 0.75)):
            fusion_notes.append("model_soft_distracted_boost")

        fatigue_score = rule_fatigue * rule_weight + model_fatigue_score * model_weight
        distraction_score = rule_distraction * rule_weight + model_distraction_score * model_weight

        if (
            probabilities["normal"] >= float(thresholds.get("normal_override_threshold", 0.90))
            and probabilities["eye_closed"] < float(thresholds.get("eye_closed_support", 0.80))
            and probabilities["yawn"] < float(thresholds.get("yawn_support", 0.65))
            and probabilities["distracted"] < float(thresholds.get("distracted_support", 0.75))
        ):
            fatigue_score *= 0.9
            distraction_score *= 0.9
            fusion_notes.append("model_normal_soft_suppression")

        fatigue_score = _clamp_score(fatigue_score)
        distraction_score = _clamp_score(distraction_score)
        overall_score = max(fatigue_score, distraction_score)
        risk_level = _to_risk_level(overall_score, risk_thresholds)

        if fatigue_score >= float(alert_cfg.get("fatigue_warning_threshold", 60.0)):
            alerts.append("fatigue_warning")
        if distraction_score >= float(alert_cfg.get("distraction_warning_threshold", 60.0)):
            alerts.append("distraction_warning")
        if risk_level in {"moderate", "severe"}:
            alerts.append(f"{risk_level}_risk")

        dedup_alerts = list(dict.fromkeys(alerts))
        base_signals["rule_fatigue_score"] = rule_fatigue
        base_signals["rule_distraction_score"] = rule_distraction
        base_signals["model_fatigue_score"] = round(model_fatigue_score, 3)
        base_signals["model_distraction_score"] = round(model_distraction_score, 3)
        base_signals["rule_weight"] = rule_weight
        base_signals["model_weight"] = model_weight
        base_signals["fusion_notes"] = fusion_notes

        return {
            "fatigue_score": round(fatigue_score, 3),
            "distraction_score": round(distraction_score, 3),
            "risk_level": risk_level,
            "signals": base_signals,
            "model_probs": probabilities,
            "alerts": dedup_alerts,
            "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
        }
