"""Offline video inference entry point for the rule-based MVP."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2

from inference.common_pipeline import CommonInferencePipeline, ensure_parent_dir, load_config, write_json_file


def _build_event_segments(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse contiguous abnormal labels into coarse event segments."""
    events: list[dict[str, Any]] = []
    active: dict[str, dict[str, Any]] = {}

    for frame in frames:
        frame_id = frame.get("frame_id")
        timestamp = frame.get("timestamp")
        labels = set(frame.get("result", {}).get("status_labels", []))
        risk_score = float(frame.get("scores", {}).get("risk_score", 0.0))

        event_labels = {label for label in labels if label not in {"normal", "no_face"}}

        for label in list(active.keys()):
            if label not in event_labels:
                active_event = active.pop(label)
                active_event["end_frame"] = frame_id
                active_event["end_time"] = timestamp
                active_event["duration_seconds"] = round(
                    max(float(timestamp) - float(active_event["start_time"]), 0.0),
                    3,
                )
                events.append(active_event)

        for label in event_labels:
            if label not in active:
                active[label] = {
                    "event_type": label,
                    "start_frame": frame_id,
                    "end_frame": frame_id,
                    "start_time": timestamp,
                    "end_time": timestamp,
                    "peak_risk_score": risk_score,
                    "peak_risk_level": frame.get("result", {}).get("risk_level", "normal"),
                }
            else:
                active[label]["end_frame"] = frame_id
                active[label]["end_time"] = timestamp
                if risk_score >= float(active[label]["peak_risk_score"]):
                    active[label]["peak_risk_score"] = risk_score
                    active[label]["peak_risk_level"] = frame.get("result", {}).get("risk_level", "normal")

    if frames:
        final_timestamp = frames[-1].get("timestamp")
        final_frame_id = frames[-1].get("frame_id")
        for active_event in active.values():
            active_event["end_frame"] = final_frame_id
            active_event["end_time"] = final_timestamp
            active_event["duration_seconds"] = round(
                max(float(final_timestamp) - float(active_event["start_time"]), 0.0),
                3,
            )
            events.append(active_event)

    return events


def run_video_inference(
    video_path: str | Path,
    config_path: str | Path,
    output_json_path: str | Path | None = None,
    output_video_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run frame-by-frame analysis on a local video and return structured JSON."""
    config = load_config(config_path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Failed to open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    writer = None
    if output_video_path:
        ensure_parent_dir(output_video_path)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_video_path), fourcc, fps if fps > 0 else 15.0, (width, height))

    pipeline = CommonInferencePipeline(config)
    frames: list[dict[str, Any]] = []
    warnings: list[str] = []
    processed_frames = 0
    failed_frames = 0
    peak_risk_score = 0.0
    peak_risk_level = "normal"

    try:
        frame_id = 0
        while True:
            success, frame = capture.read()
            if not success:
                break

            timestamp = float(frame_id / fps) if fps > 0 else float(frame_id)
            try:
                result = pipeline.process_frame(
                    frame,
                    frame_id=frame_id,
                    timestamp=timestamp,
                    draw_overlay=bool(writer),
                )
                frame_payload = result.to_dict(include_landmarks=False)
                frames.append(frame_payload)
                processed_frames += 1

                if result.risk_score >= peak_risk_score:
                    peak_risk_score = result.risk_score
                    peak_risk_level = result.risk_level

                if writer is not None:
                    annotated_frame = result.annotated_frame if result.annotated_frame is not None else frame
                    writer.write(annotated_frame)
            except Exception:
                failed_frames += 1
                warnings.append(f"frame_{frame_id}_processing_failed")

            frame_id += 1
    finally:
        capture.release()
        pipeline.close()
        if writer is not None:
            writer.release()

    average_risk = round(
        sum(frame["scores"]["risk_score"] for frame in frames) / len(frames),
        3,
    ) if frames else 0.0
    events = _build_event_segments(frames)

    payload: dict[str, Any] = {
        "status": "success" if failed_frames == 0 else "partial_success",
        "message": "video inference completed" if failed_frames == 0 else "video inference completed with frame-level failures",
        "task_type": "video_inference",
        "source": str(video_path),
        "summary": {
            "total_frames": total_frames,
            "processed_frames": processed_frames,
            "failed_frames": failed_frames,
            "fps": fps,
            "frame_size": {"width": width, "height": height},
            "average_risk_score": average_risk,
            "peak_risk_score": round(peak_risk_score, 3),
            "peak_risk_level": peak_risk_level,
            "event_count": len(events),
            "saved_visualization": bool(output_video_path),
        },
        "frames": frames,
        "events": events,
        "meta": {
            "config_path": str(config_path),
            "warnings": warnings,
        },
    }

    if output_json_path:
        write_json_file(output_json_path, payload)

    return payload


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for video inference."""
    parser = argparse.ArgumentParser(description="Run offline video inference.")
    parser.add_argument("--video", type=str, required=True, help="Path to the input video.")
    parser.add_argument("--config", type=str, default="configs/mvp.yaml", help="Path to the YAML config.")
    parser.add_argument("--output-json", type=str, default="", help="Path to save JSON result.")
    parser.add_argument("--output-video", type=str, default="", help="Path to save annotated video.")
    return parser.parse_args()


def main() -> None:
    """CLI entry point for video inference."""
    args = parse_args()
    payload = run_video_inference(
        video_path=args.video,
        config_path=args.config,
        output_json_path=args.output_json or None,
        output_video_path=args.output_video or None,
    )
    print(f"[video_infer] {payload['status']}: {payload['message']}")


if __name__ == "__main__":
    main()
