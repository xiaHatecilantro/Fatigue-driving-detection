"""Rebuild train/val split with group-aware leakage prevention.

This script reads an existing prepared dataset tree and regenerates a new split
where correlated samples stay in the same split. It is designed as a minimal
repair step for the current competition dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CLASSES = ["normal", "eye_closed", "yawn", "distracted"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(slots=True)
class SampleRecord:
    """One image sample from the existing prepared dataset."""

    source_path: Path
    filename: str
    label: str
    source_name: str
    group_key: str


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Rebuild grouped train/val split without sequence leakage.")
    parser.add_argument(
        "--input-root",
        type=str,
        default="data/processed/unified_dataset",
        help="Existing prepared dataset root.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="data/processed/unified_dataset_v2",
        help="Output dataset root for the regrouped split.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation ratio applied at the group level.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic splitting.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute the split and reports without copying files.",
    )
    return parser.parse_args()


def detect_source_name(filename: str) -> str:
    """Infer source dataset from the unified filename prefix."""
    if filename.startswith("statefarm_"):
        return "statefarm"
    if filename.startswith("uta_"):
        return "uta"
    return "unknown"


def build_group_key(filename: str, source_name: str) -> str:
    """Build a leakage-prevention group key from filename conventions.

    For UTA, keep all frames from the same sequence together.
    For StateFarm, group by approximate original image stem because the current
    filenames do not preserve driver ids. This is weaker than driver-based
    splitting but still keeps exact source-image variants together.
    """
    stem = Path(filename).stem
    if source_name == "uta":
        parts = stem.split("_")
        if len(parts) >= 6:
            return "_".join(parts[:-1])
        return stem
    if source_name == "statefarm":
        parts = stem.split("_")
        if len(parts) >= 4:
            return "_".join(parts[:-1])
        return stem
    return stem


def collect_samples(input_root: Path) -> list[SampleRecord]:
    """Scan an existing unified dataset tree and collect labeled samples."""
    samples: list[SampleRecord] = []
    for split in ("train", "val"):
        split_root = input_root / split
        if not split_root.exists():
            continue
        for label in CLASSES:
            class_root = split_root / label
            if not class_root.exists():
                continue
            for path in class_root.rglob("*"):
                if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                source_name = detect_source_name(path.name)
                group_key = build_group_key(path.name, source_name)
                samples.append(
                    SampleRecord(
                        source_path=path,
                        filename=path.name,
                        label=label,
                        source_name=source_name,
                        group_key=group_key,
                    )
                )
    if not samples:
        raise ValueError(f"No samples found under {input_root}")
    return samples


def split_by_group(
    samples: list[SampleRecord],
    val_ratio: float,
    seed: int,
) -> dict[str, list[SampleRecord]]:
    """Split samples into train/val while keeping each group in one split."""
    grouped: dict[str, dict[str, list[SampleRecord]]] = defaultdict(lambda: defaultdict(list))
    for sample in samples:
        grouped[sample.label][sample.group_key].append(sample)

    randomizer = random.Random(seed)
    result: dict[str, list[SampleRecord]] = {"train": [], "val": []}

    for label, groups in grouped.items():
        group_items = list(groups.items())
        randomizer.shuffle(group_items)

        if len(group_items) <= 1 or val_ratio <= 0.0:
            val_groups = set()
        else:
            target_val_samples = max(1, round(sum(len(items) for _, items in group_items) * val_ratio))
            selected_groups: list[str] = []
            selected_count = 0
            for group_key, items in group_items:
                if selected_count >= target_val_samples:
                    break
                selected_groups.append(group_key)
                selected_count += len(items)
            if len(selected_groups) == len(group_items):
                selected_groups = selected_groups[:-1]
            val_groups = set(selected_groups)

        for group_key, items in group_items:
            split_name = "val" if group_key in val_groups else "train"
            result[split_name].extend(items)

    return result


def exact_duplicate_groups(samples: list[SampleRecord]) -> list[list[str]]:
    """Find exact duplicate files by hash across the collected sample set."""
    by_hash: dict[str, list[str]] = defaultdict(list)
    for sample in samples:
        digest = hashlib.md5(sample.source_path.read_bytes()).hexdigest()
        by_hash[digest].append(str(sample.source_path))
    return [paths for paths in by_hash.values() if len(paths) > 1]


def build_manifest_rows(split_map: dict[str, list[SampleRecord]], output_root: Path) -> list[dict[str, Any]]:
    """Build manifest rows for the regrouped dataset."""
    rows: list[dict[str, Any]] = []
    for split_name, samples in split_map.items():
        for sample in samples:
            target_path = output_root / split_name / sample.label / sample.filename
            rows.append(
                {
                    "split": split_name,
                    "label": sample.label,
                    "source_name": sample.source_name,
                    "group_key": sample.group_key,
                    "filename": sample.filename,
                    "source_path": str(sample.source_path),
                    "target_path": str(target_path),
                }
            )
    return rows


def build_summary(
    input_root: Path,
    output_root: Path,
    split_map: dict[str, list[SampleRecord]],
    duplicates: list[list[str]],
    dry_run: bool,
) -> dict[str, Any]:
    """Build summary JSON for the regrouped split."""
    all_samples = split_map["train"] + split_map["val"]
    per_class = {
        label: {
            "total": sum(1 for sample in all_samples if sample.label == label),
            "train": sum(1 for sample in split_map["train"] if sample.label == label),
            "val": sum(1 for sample in split_map["val"] if sample.label == label),
        }
        for label in CLASSES
    }
    source_split = defaultdict(lambda: defaultdict(int))
    for split_name, samples in split_map.items():
        for sample in samples:
            source_split[sample.source_name][split_name] += 1

    group_split = defaultdict(lambda: defaultdict(set))
    for split_name, samples in split_map.items():
        for sample in samples:
            group_split[sample.label][sample.group_key].add(split_name)

    leaked_groups = {
        label: sorted(group_key for group_key, splits in groups.items() if len(splits) > 1)
        for label, groups in group_split.items()
    }

    return {
        "status": "dry_run" if dry_run else "success",
        "input_root": str(input_root),
        "output_root": str(output_root),
        "total_samples": len(all_samples),
        "per_class_stats": per_class,
        "source_split_counts": {source: dict(counts) for source, counts in source_split.items()},
        "exact_duplicate_hash_groups": len(duplicates),
        "leaked_group_keys_after_resplit": leaked_groups,
    }


def copy_split(split_map: dict[str, list[SampleRecord]], output_root: Path, dry_run: bool) -> None:
    """Copy samples into the regrouped dataset tree."""
    if dry_run:
        return

    if output_root.exists():
        shutil.rmtree(output_root)

    for split_name, samples in split_map.items():
        for sample in samples:
            target_dir = output_root / split_name / sample.label
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(sample.source_path, target_dir / sample.filename)


def write_reports(
    output_root: Path,
    summary: dict[str, Any],
    manifest_rows: list[dict[str, Any]],
    duplicates: list[list[str]],
    dry_run: bool,
) -> None:
    """Write summary and manifests for the regrouped dataset."""
    report_root = output_root / "_dry_run_reports" if dry_run else output_root
    manifests_root = report_root / "manifests"
    manifests_root.mkdir(parents=True, exist_ok=True)

    (report_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (manifests_root / "manifest.json").write_text(
        json.dumps(manifest_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (manifests_root / "duplicate_files.json").write_text(
        json.dumps(duplicates, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)

    samples = collect_samples(input_root)
    split_map = split_by_group(samples, val_ratio=args.val_ratio, seed=args.seed)
    duplicates = exact_duplicate_groups(samples)
    manifest_rows = build_manifest_rows(split_map, output_root)
    copy_split(split_map, output_root=output_root, dry_run=args.dry_run)
    summary = build_summary(
        input_root=input_root,
        output_root=output_root,
        split_map=split_map,
        duplicates=duplicates,
        dry_run=args.dry_run,
    )
    write_reports(output_root, summary, manifest_rows, duplicates, dry_run=args.dry_run)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
