"""Read-only service for exposing training metrics to the frontend."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from backend.schemas.result import TrainingMetrics, TrainingMetricsResponse


class MetricsService:
    """Load training artifacts and convert them to API response schemas."""

    def __init__(
        self,
        run_root: str | Path = "training/outputs/mobilenetv3_baseline",
        model_name: str = "MobileNetV3-Small",
    ) -> None:
        """Store artifact locations."""
        self.run_root = Path(run_root)
        self.model_name = model_name

    def get_training_metrics(self) -> TrainingMetricsResponse:
        """Load best-eval metrics and training curves from disk."""
        metrics_path = self.run_root / "metrics" / "eval_metrics.json"
        summary_path = self.run_root / "metrics" / "metrics_summary.json"
        train_log_path = self.run_root / "runs" / "train_log.jsonl"

        eval_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
        train_curve: list[dict[str, Any]] = []
        if train_log_path.exists():
            with train_log_path.open("r", encoding="utf-8") as file:
                for line in file:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    train_curve.append(
                        {
                            "epoch": int(row["epoch"]),
                            "train_loss": float(row["train_loss"]),
                            "val_loss": float(row["val_loss"]),
                            "val_accuracy": float(row["val_metrics"]["accuracy"]),
                            "val_f1": float(row["val_metrics"]["f1_macro"]),
                        }
                    )

        metrics = TrainingMetrics(**eval_payload["metrics"])
        return TrainingMetricsResponse(
            model_name=self.model_name,
            checkpoint=str(eval_payload["checkpoint"]),
            best_epoch=int(summary_payload["epoch"]),
            class_names=list(eval_payload["class_names"]),
            metrics=metrics,
            train_curve=train_curve,
        )


metrics_service = MetricsService()
