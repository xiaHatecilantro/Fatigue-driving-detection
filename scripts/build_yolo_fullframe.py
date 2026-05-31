"""Convert YOLO detection dataset to classification format for full-frame training.

Each image gets ONE label based on priority (worst state first):
  eye_closed > yawn > distracted > normal
"""

from __future__ import annotations

import shutil
from collections import Counter
from pathlib import Path

SOURCE_ROOT = Path("data/raw/driver_fatigue_yolo/datasets")
OUTPUT_ROOT = Path("data/processed/yolo_fullframe")

# YOLO class_id -> our target class
CLASS_MAP = {
    0: "normal",       # open_eyes
    1: "eye_closed",   # close_eyes
    2: "yawn",         # yawn
    3: "normal",       # mouth
    4: "distracted",   # head_low
    5: "normal",       # head_normal
    6: "distracted",   # head_rise
}
PRIORITY = ["eye_closed", "yawn", "distracted", "normal"]


def assign_label(label_dir: Path) -> str | None:
    """Read all labels for one image and return the highest-priority class."""
    targets = set()
    for line in label_dir.read_text().strip().split("\n"):
        parts = line.strip().split()
        if parts:
            targets.add(CLASS_MAP.get(int(parts[0]), "normal"))
    for label in PRIORITY:
        if label in targets:
            return label
    return None


def process_split(split: str) -> dict[str, int]:
    """Convert one split (train/val) to classification directory structure."""
    src_images = SOURCE_ROOT / split / "images"
    src_labels = SOURCE_ROOT / split / "labels"
    counts: dict[str, int] = Counter()

    for img_path in sorted(src_images.glob("*.jpg")):
        label_path = src_labels / f"{img_path.stem}.txt"
        if not label_path.exists():
            continue
        target = assign_label(label_path)
        if target is None:
            continue

        dst_dir = OUTPUT_ROOT / split / target
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img_path, dst_dir / img_path.name)
        counts[target] += 1

    return dict(counts)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict[str, int]] = {}

    for split in ["train", "val"]:
        print(f"Processing {split}...")
        counts = process_split(split)
        summary[split] = counts
        total = sum(counts.values())
        print(f"  Total: {total}")
        for cls_name, count in sorted(counts.items()):
            print(f"    {cls_name}: {count} ({count / total * 100:.1f}%)")

    print(f"\nDone. Output: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
