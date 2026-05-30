"""Training entry point for YOLO11 driver state classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import yaml
from ultralytics import YOLO


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load YAML config from disk."""
    with Path(config_path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Train YOLO11 classification model for driver state detection."
    )
    parser.add_argument(
        "--config", type=str, default="training/configs/yolo.yaml",
        help="Path to YAML config.",
    )
    parser.add_argument(
        "--device", type=str, default="",
        help="Optional device override (e.g. '0' for GPU 0, 'cpu').",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point for YOLO11 classifier training."""
    args = parse_args()
    config = load_config(args.config)
    train_cfg = config["train"]
    output_cfg = config["output"]
    data_dir = str(config["data"])
    model_name = str(config["model"]["name"])

    device = args.device or str(train_cfg.get("device", "auto"))
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model_pt = f"{model_name}.pt"
    model = YOLO(model_pt)

    results = model.train(
        data=data_dir,
        epochs=int(train_cfg.get("epochs", 12)),
        imgsz=int(train_cfg.get("imgsz", 224)),
        batch=int(train_cfg.get("batch", 32)),
        workers=int(train_cfg.get("workers", 0)),
        lr0=float(train_cfg.get("lr0", 0.0005)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0005)),
        seed=int(train_cfg.get("seed", 42)),
        device=device,
        pretrained=bool(train_cfg.get("pretrained", True)),
        optimizer=str(train_cfg.get("optimizer", "AdamW")),
        patience=int(train_cfg.get("patience", 4)),
        warmup_epochs=int(train_cfg.get("warmup_epochs", 1)),
        cos_lr=bool(train_cfg.get("cos_lr", False)),
        hsv_h=float(train_cfg.get("hsv_h", 0.015)),
        hsv_s=float(train_cfg.get("hsv_s", 0.7)),
        hsv_v=float(train_cfg.get("hsv_v", 0.4)),
        degrees=float(train_cfg.get("degrees", 8.0)),
        fliplr=float(train_cfg.get("fliplr", 0.5)),
        project=str(output_cfg["root_dir"]),
        name=str(output_cfg["run_name"]),
        save=bool(train_cfg.get("save", True)),
        exist_ok=True,
    )

    summary = {
        "model": model_name,
        "data": data_dir,
        "epochs": int(train_cfg.get("epochs", 12)),
        "class_names": ["normal", "eye_closed", "yawn", "distracted"],
        "ultralytics_class_order": list(model.names.values()),
    }
    output_root = Path(str(output_cfg["root_dir"])) / str(output_cfg["run_name"])
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Training complete. Best model: {output_root}/weights/best.pt")


if __name__ == "__main__":
    main()
