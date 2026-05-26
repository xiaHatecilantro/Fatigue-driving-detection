"""Prepare a unified classification dataset from multiple read-only sources."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(slots=True)
class CandidateSample:
    """A source sample after successful class mapping and validation."""

    source_name: str
    source_path: Path
    relative_path: Path
    target_label: str


@dataclass(slots=True)
class SkippedSample:
    """A skipped source sample with a concrete reason."""

    source_name: str
    source_path: str
    reason: str
    detail: str


def load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    """Load dataset preparation config from YAML."""
    with Path(config_path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for dataset preparation."""
    parser = argparse.ArgumentParser(description="Prepare a unified training dataset.")
    parser.add_argument(
        "--config",
        type=str,
        default="training/configs/dataset_map.yaml",
        help="Path to dataset mapping YAML config.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and summarize without writing output files.",
    )
    return parser.parse_args()


def normalize_text(text: str) -> str:
    """Normalize text for case-insensitive path matching."""
    return text.replace("\\", "/").lower()


def iter_source_files(source_root: Path, recursive: bool) -> list[Path]:
    """Collect candidate image files from a source root."""
    if recursive:
        return [path for path in source_root.rglob("*") if path.is_file()]
    return [path for path in source_root.iterdir() if path.is_file()]


def determine_label(
    relative_path: Path,
    mapping_rules: list[dict[str, Any]],
    default_label: str | None,
) -> str | None:
    """Determine target label from ordered mapping rules using relative path matching."""
    rel_text = normalize_text(str(relative_path))
    for rule in mapping_rules:
        patterns = [normalize_text(str(item)) for item in rule.get("match_any", [])]
        if patterns and any(pattern in rel_text for pattern in patterns):
            return str(rule["to"])
    return default_label


def build_flat_filename(sample: CandidateSample) -> str:
    """Build a collision-resistant output filename while preserving source traceability."""
    sanitized_rel = "__".join(sample.relative_path.parts)
    sanitized_rel = sanitized_rel.replace(" ", "_")
    return f"{sample.source_name}__{sanitized_rel}"


def write_json(output_path: Path, payload: Any) -> None:
    """Write JSON to disk with UTF-8 encoding."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def write_jsonl(output_path: Path, rows: list[dict[str, Any]]) -> None:
    """Write JSON Lines rows to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def resolve_report_root(output_root: Path, dry_run: bool) -> Path:
    """Return the directory used for reports and manifests.

    Dry-run should avoid touching the prepared dataset tree, so reports are
    written into a dedicated side directory.
    """
    return output_root / "_dry_run_reports" if dry_run else output_root


def ensure_clean_output_root(output_root: Path, dry_run: bool) -> None:
    """Remove previous prepared output tree before a fresh run."""
    if dry_run or not output_root.exists():
        return
    shutil.rmtree(output_root)


def validate_config(config: dict[str, Any]) -> None:
    """Validate required config structure early."""
    dataset_cfg = config.get("dataset", {})
    class_names = dataset_cfg.get("classes", [])
    if not class_names:
        raise ValueError("Config must define dataset.classes.")
    if len(set(class_names)) != len(class_names):
        raise ValueError("dataset.classes contains duplicate labels.")
    if "sources" not in config or not config["sources"]:
        raise ValueError("Config must define at least one source in sources.")


def collect_samples(config: dict[str, Any]) -> tuple[list[CandidateSample], list[SkippedSample], dict[str, Any]]:
    """Scan all configured sources and classify files into mapped or skipped samples."""
    dataset_cfg = config["dataset"]
    allowed_exts = {
        normalize_text(ext) if str(ext).startswith(".") else f".{normalize_text(str(ext))}"
        for ext in dataset_cfg.get("allowed_extensions", sorted(SUPPORTED_IMAGE_EXTENSIONS))
    }
    valid_classes = set(dataset_cfg["classes"])
    default_label = dataset_cfg.get("default_label")
    all_samples: list[CandidateSample] = []
    skipped: list[SkippedSample] = []
    source_summaries: dict[str, Any] = {}

    for source_cfg in config["sources"]:
        source_name = str(source_cfg["name"])
        source_root = Path(str(source_cfg["input_root"]))
        recursive = bool(source_cfg.get("recursive", True))
        exclude_patterns = [normalize_text(str(item)) for item in source_cfg.get("exclude_if_contains", [])]
        mapping_rules = list(source_cfg.get("mappings", []))
        source_default_label = source_cfg.get("default_label", default_label)

        scanned_count = 0
        mapped_count = 0
        skipped_count = 0

        if not source_root.exists():
            skipped.append(
                SkippedSample(
                    source_name=source_name,
                    source_path=str(source_root),
                    reason="missing_source_root",
                    detail="configured source root does not exist",
                )
            )
            source_summaries[source_name] = {
                "input_root": str(source_root),
                "scanned_files": 0,
                "mapped_files": 0,
                "skipped_files": 1,
            }
            continue

        for source_path in iter_source_files(source_root, recursive):
            scanned_count += 1
            relative_path = source_path.relative_to(source_root)
            rel_text = normalize_text(str(relative_path))
            extension = normalize_text(source_path.suffix)

            if extension not in allowed_exts:
                skipped.append(
                    SkippedSample(
                        source_name=source_name,
                        source_path=str(source_path),
                        reason="unsupported_extension",
                        detail=f"extension {source_path.suffix} is not allowed",
                    )
                )
                skipped_count += 1
                continue

            if any(pattern in rel_text for pattern in exclude_patterns):
                skipped.append(
                    SkippedSample(
                        source_name=source_name,
                        source_path=str(source_path),
                        reason="excluded_by_pattern",
                        detail="matched exclude_if_contains rule",
                    )
                )
                skipped_count += 1
                continue

            label = determine_label(relative_path, mapping_rules, source_default_label)
            if label is None:
                skipped.append(
                    SkippedSample(
                        source_name=source_name,
                        source_path=str(source_path),
                        reason="unmapped_label",
                        detail="no mapping rule matched and no default label provided",
                    )
                )
                skipped_count += 1
                continue

            if label not in valid_classes:
                skipped.append(
                    SkippedSample(
                        source_name=source_name,
                        source_path=str(source_path),
                        reason="invalid_target_label",
                        detail=f"mapped label '{label}' is not in dataset.classes",
                    )
                )
                skipped_count += 1
                continue

            all_samples.append(
                CandidateSample(
                    source_name=source_name,
                    source_path=source_path,
                    relative_path=relative_path,
                    target_label=str(label),
                )
            )
            mapped_count += 1

        source_summaries[source_name] = {
            "input_root": str(source_root),
            "scanned_files": scanned_count,
            "mapped_files": mapped_count,
            "skipped_files": skipped_count,
        }

    return all_samples, skipped, source_summaries


def split_samples(
    samples: list[CandidateSample],
    val_ratio: float,
    seed: int,
) -> dict[str, list[CandidateSample]]:
    """Perform deterministic per-class train/val split."""
    grouped: dict[str, list[CandidateSample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.target_label].append(sample)

    randomizer = random.Random(seed)
    split_result: dict[str, list[CandidateSample]] = {"train": [], "val": []}

    for label, items in grouped.items():
        shuffled = list(items)
        randomizer.shuffle(shuffled)

        if len(shuffled) <= 1 or val_ratio <= 0:
            val_count = 0
        else:
            val_count = max(1, int(round(len(shuffled) * val_ratio)))
            val_count = min(val_count, len(shuffled) - 1)

        split_result["val"].extend(shuffled[:val_count])
        split_result["train"].extend(shuffled[val_count:])

    return split_result


def copy_samples(
    split_samples_map: dict[str, list[CandidateSample]],
    output_root: Path,
    dry_run: bool,
) -> tuple[dict[str, dict[str, int]], list[dict[str, Any]]]:
    """Copy mapped samples into the unified output tree and emit manifest rows."""
    split_stats: dict[str, dict[str, int]] = {"train": Counter(), "val": Counter()}  # type: ignore[assignment]
    manifest_rows: list[dict[str, Any]] = []

    for split_name, samples in split_samples_map.items():
        for sample in samples:
            target_dir = output_root / split_name / sample.target_label
            target_name = build_flat_filename(sample)
            target_path = target_dir / target_name

            manifest_rows.append(
                {
                    "split": split_name,
                    "label": sample.target_label,
                    "source_name": sample.source_name,
                    "source_path": str(sample.source_path),
                    "relative_path": str(sample.relative_path),
                    "target_path": str(target_path),
                }
            )
            split_stats[split_name][sample.target_label] += 1

            if dry_run:
                continue

            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(sample.source_path, target_path)

    return split_stats, manifest_rows


def build_summary(
    config_path: str | Path,
    config: dict[str, Any],
    dry_run: bool,
    source_summaries: dict[str, Any],
    all_samples: list[CandidateSample],
    skipped: list[SkippedSample],
    split_stats: dict[str, dict[str, int]],
) -> dict[str, Any]:
    """Build summary payload for the prepared dataset run."""
    dataset_cfg = config["dataset"]
    overall_class_counts = Counter(sample.target_label for sample in all_samples)
    skipped_reason_counts = Counter(item.reason for item in skipped)
    per_class_stats = {
        label: {
            "total": int(overall_class_counts.get(label, 0)),
            "train": int(split_stats.get("train", {}).get(label, 0)),
            "val": int(split_stats.get("val", {}).get(label, 0)),
        }
        for label in dataset_cfg["classes"]
    }

    return {
        "status": "dry_run" if dry_run else "success",
        "config_path": str(config_path),
        "dry_run": dry_run,
        "dataset": {
            "output_root": str(dataset_cfg["output_root"]),
            "classes": list(dataset_cfg["classes"]),
            "val_ratio": float(dataset_cfg.get("val_ratio", 0.2)),
            "seed": int(dataset_cfg.get("seed", 42)),
        },
        "sources": source_summaries,
        "summary": {
            "mapped_total": len(all_samples),
            "skipped_total": len(skipped),
            "class_counts_total": dict(sorted(overall_class_counts.items())),
            "per_class_stats": per_class_stats,
            "split_counts": {
                split_name: dict(sorted(class_counts.items()))
                for split_name, class_counts in split_stats.items()
            },
            "skipped_reason_counts": dict(sorted(skipped_reason_counts.items())),
        },
    }


def main() -> None:
    """CLI entry point for dataset preparation."""
    args = parse_args()
    config = load_yaml_config(args.config)
    validate_config(config)

    dataset_cfg = config["dataset"]
    output_root = Path(str(dataset_cfg["output_root"]))
    val_ratio = float(dataset_cfg.get("val_ratio", 0.2))
    seed = int(dataset_cfg.get("seed", 42))

    all_samples, skipped, source_summaries = collect_samples(config)
    split_map = split_samples(all_samples, val_ratio=val_ratio, seed=seed)

    ensure_clean_output_root(output_root, dry_run=args.dry_run)
    split_stats, manifest_rows = copy_samples(split_map, output_root=output_root, dry_run=args.dry_run)

    summary = build_summary(
        config_path=args.config,
        config=config,
        dry_run=args.dry_run,
        source_summaries=source_summaries,
        all_samples=all_samples,
        skipped=skipped,
        split_stats=split_stats,
    )

    skipped_rows = [
        {
            "source_name": item.source_name,
            "source_path": item.source_path,
            "reason": item.reason,
            "detail": item.detail,
        }
        for item in skipped
    ]

    report_root = resolve_report_root(output_root, dry_run=args.dry_run)
    write_json(report_root / "summary.json", summary)
    write_json(report_root / "manifests" / "manifest.json", manifest_rows)
    write_jsonl(report_root / "manifests" / "skipped_files.jsonl", skipped_rows)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
