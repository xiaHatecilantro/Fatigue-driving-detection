"""Evaluation entry point for the lightweight driver state classifier."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from training.datasets.image_dataset import DriverStateImageDataset
from training.models.mobilenet_baseline import build_model
from training.train_classifier import compute_metrics, resolve_device


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load YAML config from disk."""
    with Path(config_path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def build_val_loader(config: dict[str, Any]) -> tuple[DataLoader[Any], list[str]]:
    """Build the validation dataloader."""
    data_cfg = config["data"]
    input_cfg = config["input"]
    eval_cfg = config["eval"]
    class_names = list(data_cfg["class_names"])

    dataset = DriverStateImageDataset(
        root_dir=data_cfg["root_dir"],
        split=str(data_cfg.get("val_split", "val")),
        class_names=class_names,
        image_size=int(input_cfg["image_size"]),
        mean=list(input_cfg["mean"]),
        std=list(input_cfg["std"]),
        augment=False,
        augmentation_config={},
        return_path=True,
        allowed_extensions=list(data_cfg.get("allowed_extensions", [])),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=int(eval_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(eval_cfg["num_workers"]),
    )
    return dataloader, class_names


def evaluate(
    model: nn.Module,
    dataloader: DataLoader[Any],
    device: torch.device,
) -> dict[str, Any]:
    """Run evaluation and return aggregate metrics."""
    model.eval()
    predictions: list[int] = []
    targets: list[int] = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            logits = model(images)
            batch_predictions = torch.argmax(logits, dim=1)
            predictions.extend(batch_predictions.cpu().tolist())
            targets.extend(labels.cpu().tolist())

    return compute_metrics(predictions, targets, num_classes=len(dataloader.dataset.class_names))


def save_confusion_matrix_csv(
    output_path: Path,
    confusion_matrix: list[list[int]],
    class_names: list[str],
) -> None:
    """Save confusion matrix CSV for later analysis."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["label"] + class_names)
        for label_name, row in zip(class_names, confusion_matrix):
            writer.writerow([label_name] + row)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Evaluate the driver state classifier baseline.")
    parser.add_argument("--config", type=str, default="training/configs/base.yaml", help="Path to YAML config.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the checkpoint to evaluate.")
    parser.add_argument("--device", type=str, default="", help="Optional device override.")
    return parser.parse_args()


def main() -> None:
    """CLI entry point for classifier evaluation."""
    args = parse_args()
    config = load_config(args.config)
    device = resolve_device(args.device or str(config["train"].get("device", "auto")))

    dataloader, class_names = build_val_loader(config)
    model = build_model(config, num_classes=len(class_names)).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    metrics = evaluate(model, dataloader, device)

    output_root = Path(str(config["output"]["root_dir"])) / str(config["output"]["run_name"])
    metrics_path = output_root / "metrics" / "eval_metrics.json"
    confusion_path = output_root / "figures" / "eval_confusion_matrix.csv"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "checkpoint": args.checkpoint,
        "class_names": class_names,
        "metrics": metrics,
    }
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    save_confusion_matrix_csv(confusion_path, metrics["confusion_matrix"], class_names)

    print(
        f"accuracy={metrics['accuracy']:.4f}, "
        f"precision={metrics['precision_macro']:.4f}, "
        f"recall={metrics['recall_macro']:.4f}, "
        f"f1={metrics['f1_macro']:.4f}"
    )


if __name__ == "__main__":
    main()
