import { useEffect, useState } from "react";
import ResultCard from "./components/ResultCard";
import TrainingMetricsPanel from "./components/TrainingMetricsPanel";
import UploadPanel from "./components/UploadPanel";
import {
  getHealth,
  getTrainingMetrics,
  inferImage,
  inferVideo,
} from "./services/api";
import type {
  ImageInferenceResponse,
  TrainingMetricsResponse,
  VideoInferenceResponse,
} from "./types/api";

export default function App() {
  const [health, setHealth] = useState<string>("检测中");
  const [trainingMetrics, setTrainingMetrics] =
    useState<TrainingMetricsResponse | null>(null);
  const [imageResponse, setImageResponse] = useState<ImageInferenceResponse | null>(null);
  const [videoResponse, setVideoResponse] = useState<VideoInferenceResponse | null>(null);
  const [imageBusy, setImageBusy] = useState(false);
  const [videoBusy, setVideoBusy] = useState(false);
  const [globalError, setGlobalError] = useState("");

  useEffect(() => {
    getHealth()
      .then((payload) => {
        setHealth(`${payload.status} · ${payload.service} · v${payload.version}`);
      })
      .catch((error: unknown) => {
        setHealth("后端不可用");
        setGlobalError(error instanceof Error ? error.message : "健康检查失败");
      });

    getTrainingMetrics()
      .then((payload) => {
        setTrainingMetrics(payload);
      })
      .catch(() => {
        // Keep the page usable even when metrics are not available.
      });
  }, []);

  async function handleImageInference(file: File) {
    setImageBusy(true);
    setGlobalError("");
    try {
      const payload = await inferImage(file, false);
      setImageResponse(payload);
    } catch (error) {
      setGlobalError(error instanceof Error ? error.message : "图片推理失败");
    } finally {
      setImageBusy(false);
    }
  }

  async function handleVideoInference(file: File) {
    setVideoBusy(true);
    setGlobalError("");
    try {
      const payload = await inferVideo(file, false, true);
      setVideoResponse(payload);
    } catch (error) {
      setGlobalError(error instanceof Error ? error.message : "视频推理失败");
    } finally {
      setVideoBusy(false);
    }
  }

  return (
    <div className="app-shell">
      <header className="hero">
        <div>
          <p className="eyebrow">驾驶员监测演示系统</p>
          <h1>驾驶员疲劳与分心检测系统</h1>
          <p className="hero-copy">
            联调规则法、轻量分类模型与融合服务。当前前端保留图片推理和视频推理两条离线检测链路。
          </p>
        </div>
        <div className="health-card">
          <span>后端服务</span>
          <strong>{health}</strong>
        </div>
      </header>

      {globalError ? <div className="global-error">{globalError}</div> : null}

      <main className="main-grid single-pane">
        <div className="left-column">
          <TrainingMetricsPanel data={trainingMetrics} />

          <UploadPanel
            title="图片推理"
            accept="image/*"
            busy={imageBusy}
            onSubmit={handleImageInference}
          />
          <ResultCard
            title="图片推理结果"
            result={imageResponse?.result ?? null}
          />

          <UploadPanel
            title="视频推理"
            accept="video/*"
            busy={videoBusy}
            onSubmit={handleVideoInference}
          />
          <section className="panel">
            <div className="panel-header">
              <h3>视频推理摘要</h3>
            </div>
            {!videoResponse ? (
              <div className="empty-state">暂无视频结果</div>
            ) : (
              <div className="metric-grid">
                <div className="metric-tile">
                  <span>处理成功帧数</span>
                  <strong>{videoResponse.summary.processed_frames}</strong>
                </div>
                <div className="metric-tile">
                  <span>处理失败帧数</span>
                  <strong>{videoResponse.summary.failed_frames}</strong>
                </div>
                <div className="metric-tile">
                  <span>峰值风险等级</span>
                  <strong>{videoResponse.summary.peak_risk_level}</strong>
                </div>
                <div className="metric-tile">
                  <span>平均风险分</span>
                  <strong>{videoResponse.summary.average_risk_score.toFixed(2)}</strong>
                </div>
              </div>
            )}
          </section>
        </div>
      </main>
    </div>
  );
}
