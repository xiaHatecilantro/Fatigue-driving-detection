"""Minimal API tests for backend image inference."""

from __future__ import annotations

import tempfile
import unittest
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

from backend.main import app


class TestImageInferenceAPI(unittest.TestCase):
    """Validate the image inference endpoint returns the unified schema."""

    @classmethod
    def setUpClass(cls) -> None:
        """Create one shared test client."""
        cls.client = TestClient(app)

    def _build_test_image_bytes(self) -> bytes:
        """Create a simple JPEG image in memory."""
        image = np.zeros((224, 224, 3), dtype=np.uint8)
        cv2.putText(image, "test", (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
        success, buffer = cv2.imencode(".jpg", image)
        self.assertTrue(success)
        return buffer.tobytes()

    def _build_test_video_bytes(self) -> bytes:
        """Create a short MP4 video in a temporary file and return its bytes."""
        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = Path(temp_dir) / "test.mp4"
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                5.0,
                (224, 224),
            )
            self.assertTrue(writer.isOpened())

            for frame_index in range(5):
                frame = np.zeros((224, 224, 3), dtype=np.uint8)
                cv2.putText(
                    frame,
                    f"f{frame_index}",
                    (60, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                writer.write(frame)

            writer.release()
            return video_path.read_bytes()

    def test_health_endpoint(self) -> None:
        """Health endpoint should return service metadata."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "ok")
        self.assertIn("service", payload)
        self.assertIn("version", payload)

    def test_image_inference_endpoint_returns_unified_result(self) -> None:
        """Image inference should return the unified JSON result shape."""
        image_bytes = self._build_test_image_bytes()
        response = self.client.post(
            "/api/infer/image",
            files={"file": ("test.jpg", BytesIO(image_bytes), "image/jpeg")},
            data={"save_visualization": "false"},
        )
        self.assertEqual(response.status_code, 200, msg=response.text)

        payload = response.json()
        self.assertEqual(payload["status"], "success")
        self.assertIn("result", payload)

        result = payload["result"]
        for field in (
            "fatigue_score",
            "distraction_score",
            "risk_level",
            "signals",
            "model_probs",
            "alerts",
            "timestamp",
            "rule_result",
            "model_result",
            "fusion_result",
        ):
            self.assertIn(field, result)

        self.assertIsInstance(result["signals"], dict)
        self.assertIsInstance(result["model_probs"], dict)
        self.assertIsInstance(result["alerts"], list)

    def test_video_inference_endpoint_returns_summary(self) -> None:
        """Video inference should return summary information and optional frame results."""
        video_bytes = self._build_test_video_bytes()
        response = self.client.post(
            "/api/infer/video",
            files={"file": ("test.mp4", BytesIO(video_bytes), "video/mp4")},
            data={"save_visualization": "false", "include_frames": "true"},
        )
        self.assertEqual(response.status_code, 200, msg=response.text)

        payload = response.json()
        self.assertIn(payload["status"], {"success", "partial_success"})
        self.assertIn("summary", payload)
        self.assertIn("results", payload)
        self.assertIn("events", payload)

        summary = payload["summary"]
        for field in (
            "total_frames",
            "processed_frames",
            "failed_frames",
            "fps",
            "average_risk_score",
            "peak_risk_score",
            "peak_risk_level",
            "event_count",
            "saved_visualization",
        ):
            self.assertIn(field, summary)

        self.assertIsInstance(payload["results"], list)
        self.assertIsInstance(payload["events"], list)

        if payload["results"]:
            first_result = payload["results"][0]
            for field in (
                "fatigue_score",
                "distraction_score",
                "risk_level",
                "signals",
                "model_probs",
                "alerts",
                "timestamp",
                "rule_result",
                "model_result",
                "fusion_result",
            ):
                self.assertIn(field, first_result)

if __name__ == "__main__":
    unittest.main()
