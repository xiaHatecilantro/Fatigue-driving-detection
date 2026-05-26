"""Build a face-ROI version of the unified dataset for better model generalization."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Generate a face-ROI dataset from a split image dataset.")
    parser.add_argument("--input-root", type=str, default="data/processed/unified_dataset_v2")
    parser.add_argument("--output-root", type=str, default="data/processed/unified_dataset_roi_v1")
    parser.add_argument("--margin-x", type=float, default=0.18)
    parser.add_argument("--margin-y", type=float, default=0.22)
    parser.add_argument("--min-size", type=int, default=48)
    parser.add_argument("--copy-on-fail", action="store_true")
    return parser.parse_args()


def compute_face_bbox(
    image_shape: tuple[int, int, int],
    landmarks: list[tuple[float, float]],
    margin_x: float,
    margin_y: float,
) -> tuple[int, int, int, int] | None:
    """Compute a clamped face bounding box from landmarks."""
    if not landmarks:
        return None

    image_height, image_width = image_shape[:2]
    min_x = min(point[0] for point in landmarks)
    max_x = max(point[0] for point in landmarks)
    min_y = min(point[1] for point in landmarks)
    max_y = max(point[1] for point in landmarks)

    width = max_x - min_x
    height = max_y - min_y
    if width <= 1 or height <= 1:
        return None

    x1 = max(0, int(min_x - width * margin_x))
    y1 = max(0, int(min_y - height * margin_y))
    x2 = min(image_width, int(max_x + width * margin_x))
    y2 = min(image_height, int(max_y + height * margin_y))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def extract_landmarks(face_mesh: Any, image: Any) -> list[tuple[float, float]]:
    """Extract 2D image-space face landmarks."""
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb_image)
    if not result.multi_face_landmarks:
        return []

    image_height, image_width = image.shape[:2]
    return [
        (landmark.x * image_width, landmark.y * image_height)
        for landmark in result.multi_face_landmarks[0].landmark
    ]


def build_roi_dataset(
    input_root: Path,
    output_root: Path,
    margin_x: float,
    margin_y: float,
    min_size: int,
    copy_on_fail: bool,
) -> dict[str, Any]:
    """Generate a new dataset by cropping face ROI from each image."""
    output_root.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "margin_x": margin_x,
        "margin_y": margin_y,
        "min_size": min_size,
        "copy_on_fail": copy_on_fail,
        "processed": 0,
        "cropped": 0,
        "copied_on_fail": 0,
        "skipped": 0,
        "splits": {},
    }

    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.35,
        min_tracking_confidence=0.35,
    ) as face_mesh:
        for split_dir in sorted(input_root.iterdir()):
            if not split_dir.is_dir():
                continue
            split_stats: dict[str, Any] = {"processed": 0, "cropped": 0, "copied_on_fail": 0, "skipped": 0, "classes": {}}

            for class_dir in sorted(split_dir.iterdir()):
                if not class_dir.is_dir():
                    continue
                target_dir = output_root / split_dir.name / class_dir.name
                target_dir.mkdir(parents=True, exist_ok=True)
                class_stats = {"processed": 0, "cropped": 0, "copied_on_fail": 0, "skipped": 0}

                for image_path in sorted(class_dir.iterdir()):
                    if not image_path.is_file() or image_path.suffix.lower() not in VALID_EXTENSIONS:
                        continue

                    summary["processed"] += 1
                    split_stats["processed"] += 1
                    class_stats["processed"] += 1

                    image = cv2.imread(str(image_path))
                    if image is None:
                        summary["skipped"] += 1
                        split_stats["skipped"] += 1
                        class_stats["skipped"] += 1
                        continue

                    landmarks = extract_landmarks(face_mesh, image)
                    bbox = compute_face_bbox(image.shape, landmarks, margin_x=margin_x, margin_y=margin_y)
                    target_path = target_dir / image_path.name
                    wrote_file = False

                    if bbox is not None:
                        x1, y1, x2, y2 = bbox
                        roi = image[y1:y2, x1:x2]
                        if roi.size > 0 and roi.shape[0] >= min_size and roi.shape[1] >= min_size:
                            wrote_file = bool(cv2.imwrite(str(target_path), roi))
                            if wrote_file:
                                summary["cropped"] += 1
                                split_stats["cropped"] += 1
                                class_stats["cropped"] += 1

                    if not wrote_file:
                        if copy_on_fail:
                            shutil.copy2(image_path, target_path)
                            summary["copied_on_fail"] += 1
                            split_stats["copied_on_fail"] += 1
                            class_stats["copied_on_fail"] += 1
                        else:
                            summary["skipped"] += 1
                            split_stats["skipped"] += 1
                            class_stats["skipped"] += 1

                split_stats["classes"][class_dir.name] = class_stats
            summary["splits"][split_dir.name] = split_stats

    return summary


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)

    summary = build_roi_dataset(
        input_root=input_root,
        output_root=output_root,
        margin_x=float(args.margin_x),
        margin_y=float(args.margin_y),
        min_size=int(args.min_size),
        copy_on_fail=bool(args.copy_on_fail),
    )

    with (output_root / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
