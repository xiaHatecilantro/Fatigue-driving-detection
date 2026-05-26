import type {
  ImageInferenceResponse,
  TrainingMetricsResponse,
  VideoInferenceResponse,
} from "../types/api";

const API_BASE = "http://127.0.0.1:8000";

async function parseJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    throw new Error(`Request failed with status ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export async function inferImage(
  file: File,
  saveVisualization = false
): Promise<ImageInferenceResponse> {
  const form = new FormData();
  form.append("file", file);
  form.append("save_visualization", String(saveVisualization));

  const response = await fetch(`${API_BASE}/api/infer/image`, {
    method: "POST",
    body: form,
  });
  return parseJson<ImageInferenceResponse>(response);
}

export async function inferVideo(
  file: File,
  saveVisualization = false,
  includeFrames = true
): Promise<VideoInferenceResponse> {
  const form = new FormData();
  form.append("file", file);
  form.append("save_visualization", String(saveVisualization));
  form.append("include_frames", String(includeFrames));

  const response = await fetch(`${API_BASE}/api/infer/video`, {
    method: "POST",
    body: form,
  });
  return parseJson<VideoInferenceResponse>(response);
}

export async function getHealth(): Promise<{
  status: string;
  service: string;
  version: string;
}> {
  const response = await fetch(`${API_BASE}/health`);
  return parseJson(response);
}

export async function getTrainingMetrics(): Promise<TrainingMetricsResponse> {
  const response = await fetch(`${API_BASE}/api/metrics/training`);
  return parseJson<TrainingMetricsResponse>(response);
}

export function getWsUrl(): string {
  return "ws://127.0.0.1:8000/ws/realtime";
}
