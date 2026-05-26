"""Training entry point for the lightweight driver state classifier."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from training.datasets.image_dataset import DriverStateImageDataset
from training.models.mobilenet_baseline import build_model


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load YAML config from disk."""
    with Path(config_path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def set_seed(seed: int) -> None:
    """Set global random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    """Resolve the target device from config or CLI override."""
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def ensure_output_dirs(output_root: Path) -> dict[str, Path]:
    """Create standard output subdirectories."""
    paths = {
        "root": output_root,
        "runs": output_root / "runs",
        "checkpoints": output_root / "checkpoints",
        "metrics": output_root / "metrics",
        "figures": output_root / "figures",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def build_dataloaders(config: dict[str, Any]) -> tuple[DataLoader[Any], DataLoader[Any], list[str]]:
    """Build train and validation dataloaders from config."""
    data_cfg = config["data"]
    input_cfg = config["input"]
    train_cfg = config["train"]
    eval_cfg = config["eval"]
    class_names = list(data_cfg["class_names"])

    train_dataset = DriverStateImageDataset(
        root_dir=data_cfg["root_dir"],
        split=str(data_cfg.get("train_split", "train")),
        class_names=class_names,
        image_size=int(input_cfg["image_size"]),
        mean=list(input_cfg["mean"]),
        std=list(input_cfg["std"]),
        augment=True,
        augmentation_config=dict(input_cfg.get("train_augmentation", {})),
        return_path=True,
        allowed_extensions=list(data_cfg.get("allowed_extensions", [])),
    )
    val_dataset = DriverStateImageDataset(
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

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(train_cfg["num_workers"]),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(eval_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(eval_cfg["num_workers"]),
    )
    return train_loader, val_loader, class_names


def compute_metrics(
    predictions: list[int],
    targets: list[int],
    num_classes: int,
) -> dict[str, Any]:
    """Compute accuracy, macro metrics, per-class metrics, and confusion matrix."""
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for target, prediction in zip(targets, predictions):
        confusion[target, prediction] += 1

    total = int(confusion.sum())
    correct = int(np.trace(confusion))
    accuracy = correct / total if total else 0.0

    per_class_precision: list[float] = []
    per_class_recall: list[float] = []
    per_class_f1: list[float] = []
    for class_index in range(num_classes):
        true_positive = float(confusion[class_index, class_index])
        predicted_positive = float(confusion[:, class_index].sum())
        actual_positive = float(confusion[class_index, :].sum())
        precision = true_positive / predicted_positive if predicted_positive else 0.0
        recall = true_positive / actual_positive if actual_positive else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        per_class_precision.append(precision)
        per_class_recall.append(recall)
        per_class_f1.append(f1)

    return {
        "accuracy": accuracy,
        "precision_macro": float(np.mean(per_class_precision)) if per_class_precision else 0.0,
        "recall_macro": float(np.mean(per_class_recall)) if per_class_recall else 0.0,
        "f1_macro": float(np.mean(per_class_f1)) if per_class_f1 else 0.0,
        "precision_per_class": per_class_precision,
        "recall_per_class": per_class_recall,
        "f1_per_class": per_class_f1,
        "confusion_matrix": confusion.tolist(),
    }


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader[Any],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, list[int], list[int]]:
    """Run one train or validation epoch."""
    is_train = optimizer is not None
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    predictions: list[int] = []
    targets: list[int] = []

    with torch.set_grad_enabled(is_train):
        for batch in dataloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += float(loss.item()) * images.size(0)
            batch_predictions = torch.argmax(logits, dim=1)
            predictions.extend(batch_predictions.detach().cpu().tolist())
            targets.extend(labels.detach().cpu().tolist())

    dataset_size = len(dataloader.dataset)
    average_loss = total_loss / dataset_size if dataset_size else 0.0
    return average_loss, predictions, targets


def save_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, Any],
    class_names: list[str],
    config: dict[str, Any],
) -> None:
    """Save a reusable training checkpoint."""
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "class_names": class_names,
            "config": config,
        },
        checkpoint_path,
    )


def save_confusion_matrix_csv(
    output_path: Path,
    confusion_matrix: list[list[int]],
    class_names: list[str],
) -> None:
    """Save confusion matrix as CSV."""
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["label"] + class_names)
        for label_name, row in zip(class_names, confusion_matrix):
            writer.writerow([label_name] + row)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
) -> ReduceLROnPlateau | None:
    """Create an optional learning-rate scheduler."""
    scheduler_cfg = dict(config["train"].get("lr_scheduler", {}))
    if not scheduler_cfg.get("enabled", False):
        return None
    return ReduceLROnPlateau(
        optimizer,
        mode=str(scheduler_cfg.get("mode", "max")),
        factor=float(scheduler_cfg.get("factor", 0.5)),
        patience=int(scheduler_cfg.get("patience", 2)),
        min_lr=float(scheduler_cfg.get("min_lr", 1e-6)),
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Train the driver state classifier baseline.")
    parser.add_argument("--config", type=str, default="training/configs/base.yaml", help="Path to YAML config.")
    parser.add_argument("--device", type=str, default="", help="Optional device override.")
    return parser.parse_args()


def main() -> None:
    """CLI entry point for classifier training."""
    args = parse_args()
    config = load_config(args.config)
    train_cfg = config["train"]
    output_cfg = config["output"]

    set_seed(int(train_cfg["seed"]))
    device = resolve_device(args.device or str(train_cfg.get("device", "auto")))
    output_dirs = ensure_output_dirs(Path(str(output_cfg["root_dir"])) / str(output_cfg["run_name"]))

    train_loader, val_loader, class_names = build_dataloaders(config)
    model = build_model(config, num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        params=model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    scheduler = build_scheduler(optimizer, config)

    best_metric = float("-inf")
    best_metric_name = str(train_cfg.get("save_best_by", "f1_macro"))
    early_stopping_patience = int(train_cfg.get("early_stopping_patience", 0))
    epochs_without_improvement = 0
    train_log_path = output_dirs["runs"] / "train_log.jsonl"
    metrics_summary_path = output_dirs["metrics"] / "metrics_summary.json"
    confusion_path = output_dirs["figures"] / "confusion_matrix.csv"

    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        train_loss, train_predictions, train_targets = run_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        val_loss, val_predictions, val_targets = run_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            optimizer=None,
            device=device,
        )

        train_metrics = compute_metrics(train_predictions, train_targets, num_classes=len(class_names))
        val_metrics = compute_metrics(val_predictions, val_targets, num_classes=len(class_names))
        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        }

        with train_log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(epoch_record, ensure_ascii=False) + "\n")

        current_metric = float(val_metrics.get(best_metric_name, val_metrics["f1_macro"]))
        if scheduler is not None:
            scheduler.step(current_metric)
        save_checkpoint(
            checkpoint_path=output_dirs["checkpoints"] / "last.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics=epoch_record,
            class_names=class_names,
            config=config,
        )
        if current_metric >= best_metric:
            best_metric = current_metric
            epochs_without_improvement = 0
            save_checkpoint(
                checkpoint_path=output_dirs["checkpoints"] / "best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=epoch_record,
                class_names=class_names,
                config=config,
            )
            with metrics_summary_path.open("w", encoding="utf-8") as metrics_file:
                json.dump(epoch_record, metrics_file, ensure_ascii=False, indent=2)
            save_confusion_matrix_csv(confusion_path, val_metrics["confusion_matrix"], class_names)
        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"val_f1={val_metrics['f1_macro']:.4f} | "
            f"lr={optimizer.param_groups[0]['lr']:.6f}"
        )

        if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
            print(
                f"Early stopping triggered after {epoch} epochs "
                f"(no improvement for {epochs_without_improvement} epochs)."
            )
            break


if __name__ == "__main__":
    main()
