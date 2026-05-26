"""Lightweight classification model runner for inference-time enhancement."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from torchvision import transforms

from training.models.mobilenet_baseline import build_model


@dataclass(slots=True)
class ModelInferenceResult:
    """Structured model output for a single image."""

    predicted_label: str
    predicted_index: int
    probabilities: dict[str, float]


class ClassificationModelRunner:
    """Wrapper around the lightweight classifier checkpoint for single-image inference."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        config_path: str | Path | None = None,
        device: str = "auto",
    ) -> None:
        """Load checkpoint, config, transforms, and model weights."""
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        self.device = self._resolve_device(device)
        self.checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.config = self._load_config(config_path)
        self.class_names = self._load_class_names()
        self.transform = self._build_transform()
        self.model = self._build_model()
        self.model.eval()

    def _resolve_device(self, device_name: str) -> torch.device:
        """Resolve target device for model inference."""
        if device_name == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device_name)

    def _load_config(self, config_path: str | Path | None) -> dict[str, Any]:
        """Load config from explicit path or embedded checkpoint config."""
        if config_path is not None:
            with Path(config_path).open("r", encoding="utf-8") as file:
                return yaml.safe_load(file)

        embedded_config = self.checkpoint.get("config")
        if embedded_config is None:
            raise ValueError("No config provided and checkpoint does not contain embedded config.")
        return embedded_config

    def _load_class_names(self) -> list[str]:
        """Load class names from checkpoint or config."""
        checkpoint_classes = self.checkpoint.get("class_names")
        if checkpoint_classes:
            return list(checkpoint_classes)

        config_classes = self.config.get("data", {}).get("class_names", [])
        if not config_classes:
            raise ValueError("Class names missing from checkpoint and config.")
        return list(config_classes)

    def _build_transform(self) -> transforms.Compose:
        """Build validation-time normalization transform."""
        input_cfg = self.config.get("input", {})
        image_size = int(input_cfg.get("image_size", 224))
        mean = list(input_cfg.get("mean", [0.485, 0.456, 0.406]))
        std = list(input_cfg.get("std", [0.229, 0.224, 0.225]))
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    def _build_model(self) -> torch.nn.Module:
        """Build model instance and restore weights."""
        model = build_model(self.config, num_classes=len(self.class_names)).to(self.device)
        model.load_state_dict(self.checkpoint["model_state_dict"])
        return model

    def predict_from_image(self, image: Image.Image) -> ModelInferenceResult:
        """Run inference on a PIL image and return class probabilities."""
        rgb_image = image.convert("RGB")
        input_tensor = self.transform(rgb_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().tolist()

        probabilities = {
            class_name: float(probability)
            for class_name, probability in zip(self.class_names, probs)
        }
        predicted_index = int(np.argmax(probs))
        return ModelInferenceResult(
            predicted_label=self.class_names[predicted_index],
            predicted_index=predicted_index,
            probabilities=probabilities,
        )

    def predict_from_bgr_frame(self, frame: np.ndarray) -> ModelInferenceResult:
        """Run inference on an OpenCV BGR frame."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb_frame)
        return self.predict_from_image(image)

    def predict_from_path(self, image_path: str | Path) -> ModelInferenceResult:
        """Run inference from an image file path."""
        image = Image.open(image_path)
        return self.predict_from_image(image)
