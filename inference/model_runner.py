"""YOLO11 classification model runner for inference-time enhancement."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


@dataclass(slots=True)
class ModelInferenceResult:
    """Structured model output for a single image."""

    predicted_label: str
    predicted_index: int
    probabilities: dict[str, float]


# ultralytics sorts class directories alphabetically: distracted, eye_closed, normal, yawn
# Our canonical order: normal=0, eye_closed=1, yawn=2, distracted=3
_ULTRA_ORDER = ["distracted", "eye_closed", "normal", "yawn"]
_CANONICAL_ORDER = ["normal", "eye_closed", "yawn", "distracted"]

# Precompute index mapping: canonical_index -> ultra_index
_ULTRA_TO_CANONICAL = {name: _CANONICAL_ORDER.index(name) for name in _ULTRA_ORDER}


class ClassificationModelRunner:
    """Wrapper around YOLO11 classification checkpoint for single-image inference."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        config_path: str | Path | None = None,
        device: str = "auto",
    ) -> None:
        """Load YOLO11 classification checkpoint."""
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        self.model = YOLO(str(self.checkpoint_path))
        self._ultra_names = dict(self.model.names)

        if len(self._ultra_names) != len(_CANONICAL_ORDER):
            raise ValueError(
                f"Expected {len(_CANONICAL_ORDER)} classes, got {len(self._ultra_names)}: {self._ultra_names}"
            )

    def predict_from_bgr_frame(self, frame: np.ndarray) -> ModelInferenceResult:
        """Run inference on an OpenCV BGR frame.

        ultralytics YOLO handles BGR natively (OpenCV format) — no colour conversion needed.
        """
        results = self.model(frame, verbose=False)
        probs = results[0].probs

        if probs is None:
            return ModelInferenceResult(
                predicted_label="normal",
                predicted_index=0,
                probabilities={name: 0.0 for name in _CANONICAL_ORDER},
            )

        raw = probs.data.cpu().numpy() if probs.data is not None else np.zeros(len(_CANONICAL_ORDER))
        probabilities = {
            name: float(raw[_ULTRA_ORDER.index(name)])
            for name in _CANONICAL_ORDER
        }
        predicted_index = int(np.argmax([probabilities[n] for n in _CANONICAL_ORDER]))
        return ModelInferenceResult(
            predicted_label=_CANONICAL_ORDER[predicted_index],
            predicted_index=predicted_index,
            probabilities=probabilities,
        )

    def predict_from_image(self, image) -> ModelInferenceResult:
        """Run inference on a PIL image."""
        bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        return self.predict_from_bgr_frame(bgr)

    def predict_from_path(self, image_path: str | Path) -> ModelInferenceResult:
        """Run inference from an image file path."""
        bgr = cv2.imread(str(image_path))
        if bgr is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")
        return self.predict_from_bgr_frame(bgr)
