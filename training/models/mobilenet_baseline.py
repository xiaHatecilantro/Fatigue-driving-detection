"""MobileNetV3 baseline model for driver state classification."""

from __future__ import annotations

from typing import Any

import torch.nn as nn
from torchvision import models


class MobileNetV3Baseline(nn.Module):
    """Lightweight four-class classifier built on MobileNetV3-Small."""

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        dropout: float = 0.2,
        freeze_backbone: bool = False,
    ) -> None:
        """Initialize the MobileNetV3 backbone and replace the classifier head."""
        super().__init__()
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        self.backbone = models.mobilenet_v3_small(weights=weights)

        in_features = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.Hardswish(),
            nn.Dropout(p=dropout),
            nn.Linear(1024, num_classes),
        )

        if freeze_backbone:
            for parameter in self.backbone.features.parameters():
                parameter.requires_grad = False

    def forward(self, inputs: Any) -> Any:
        """Run a forward pass."""
        return self.backbone(inputs)


def build_model(config: dict[str, Any], num_classes: int) -> MobileNetV3Baseline:
    """Build the configured MobileNetV3 baseline model."""
    model_cfg = config.get("model", {})
    model_name = str(model_cfg.get("name", "mobilenet_v3_small"))
    if model_name != "mobilenet_v3_small":
        raise ValueError(f"Unsupported model name: {model_name}")

    return MobileNetV3Baseline(
        num_classes=num_classes,
        pretrained=bool(model_cfg.get("pretrained", True)),
        dropout=float(model_cfg.get("dropout", 0.2)),
        freeze_backbone=bool(model_cfg.get("freeze_backbone", False)),
    )
