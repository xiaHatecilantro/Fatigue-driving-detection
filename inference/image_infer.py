"""Offline single-image inference entry point for the rule-based MVP."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2

from inference.common_pipeline import CommonInferencePipeline, load_config, write_json_file


def run_image_inference(
    image_path: str | Path,
    config_path: str | Path,
    output_json_path: str | Path | None = None,
    output_image_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run rule-based inference on a single image and optionally save outputs."""
    config = load_config(config_path)
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")

    pipeline = CommonInferencePipeline(config)
    try:
        result = pipeline.process_frame(image, frame_id=0, timestamp=0.0, draw_overlay=bool(output_image_path))
    finally:
        pipeline.close()

    payload: dict[str, Any] = {
        "status": "success",
        "message": "image inference completed",
        "task_type": "image_inference",
        "source": str(image_path),
        "data": result.to_dict(include_landmarks=True),
        "meta": {
            "saved_visualization": bool(output_image_path),
            "saved_json": bool(output_json_path),
        },
    }

    if output_image_path:
        Path(output_image_path).parent.mkdir(parents=True, exist_ok=True)
        image_to_save = result.annotated_frame if result.annotated_frame is not None else image
        if not cv2.imwrite(str(output_image_path), image_to_save):
            payload["status"] = "partial_success"
            payload["message"] = "image inference completed, but visualization save failed"
            payload.setdefault("warnings", []).append("failed_to_save_visualization")

    if output_json_path:
        write_json_file(output_json_path, payload)

    return payload


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for image inference."""
    parser = argparse.ArgumentParser(description="Run offline image inference.")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--config", type=str, default="configs/mvp.yaml", help="Path to the YAML config.")
    parser.add_argument("--output-json", type=str, default="", help="Path to save JSON result.")
    parser.add_argument("--output-image", type=str, default="", help="Path to save annotated image.")
    return parser.parse_args()


def main() -> None:
    """CLI entry point for image inference."""
    args = parse_args()
    payload = run_image_inference(
        image_path=args.image,
        config_path=args.config,
        output_json_path=args.output_json or None,
        output_image_path=args.output_image or None,
    )
    print(f"[image_infer] {payload['status']}: {payload['message']}")


if __name__ == "__main__":
    main()
