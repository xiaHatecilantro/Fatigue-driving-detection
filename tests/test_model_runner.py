"""Tests for the YOLO11 ClassificationModelRunner interface."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from inference.model_runner import ClassificationModelRunner, ModelInferenceResult

CANONICAL_KEYS = {"normal", "eye_closed", "yawn", "distracted"}


def test_result_has_correct_keys() -> None:
    """ModelInferenceResult.probabilities must contain exactly 4 canonical keys."""
    result = ModelInferenceResult(
        predicted_label="normal",
        predicted_index=0,
        probabilities={"normal": 0.9, "eye_closed": 0.05, "yawn": 0.03, "distracted": 0.02},
    )
    assert set(result.probabilities.keys()) == CANONICAL_KEYS
    for v in result.probabilities.values():
        assert 0.0 <= v <= 1.0


def test_result_fields_consistent() -> None:
    """Predicted_label and predicted_index must match the max-probability class."""
    order = ["normal", "eye_closed", "yawn", "distracted"]
    probs = {"normal": 0.1, "eye_closed": 0.7, "yawn": 0.1, "distracted": 0.1}
    idx = max(range(4), key=lambda i: probs[order[i]])
    label = order[idx]
    result = ModelInferenceResult(predicted_label=label, predicted_index=idx, probabilities=probs)
    assert result.predicted_label == "eye_closed"
    assert result.predicted_index == 1


def test_dummy_frame_prediction() -> None:
    """ClassificationModelRunner.predict_from_bgr_frame returns valid structure on dummy input."""
    model_path = Path(
        "runs/classify/training/outputs/yolo11m_newdata_baseline/weights/best.pt"
    )
    if not model_path.exists():
        pytest.skip("Model checkpoint not found")

    runner = ClassificationModelRunner(str(model_path))
    frame = np.zeros((224, 224, 3), dtype=np.uint8)
    result = runner.predict_from_bgr_frame(frame)

    assert isinstance(result, ModelInferenceResult)
    assert set(result.probabilities.keys()) == CANONICAL_KEYS
    assert result.predicted_label in CANONICAL_KEYS
    assert 0 <= result.predicted_index <= 3
