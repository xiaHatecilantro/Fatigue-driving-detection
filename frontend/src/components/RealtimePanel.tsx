import { useEffect, useRef, useState } from "react";
import { useRealtimeInference } from "../hooks/useRealtimeInference";
import ResultCard from "./ResultCard";

function frameToBase64(video: HTMLVideoElement): string | null {
  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth || 320;
  canvas.height = video.videoHeight || 240;
  const context = canvas.getContext("2d");
  if (!context) {
    return null;
  }
  context.drawImage(video, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL("image/jpeg", 0.7);
}

function getRiskLevelText(riskLevel?: string): string {
  switch (riskLevel) {
    case "mild":
      return "轻度";
    case "moderate":
      return "中度";
    case "severe":
      return "重度";
    case "normal":
    default:
      return "正常";
  }
}

export default function RealtimePanel() {
  const { connected, lastResponse, error, connect, disconnect, sendFrame } =
    useRealtimeInference();
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const timerRef = useRef<number | null>(null);
  const [running, setRunning] = useState(false);
  const [cameraError, setCameraError] = useState("");
  const faceDetected = Boolean(lastResponse?.result?.signals?.face_detected);
  const hasRealtimeResult = Boolean(lastResponse?.result);
  const riskLevelText = getRiskLevelText(lastResponse?.result?.risk_level);
  const ear = lastResponse?.result?.signals?.ear;
  const mar = lastResponse?.result?.signals?.mar;
  const yaw = lastResponse?.result?.signals?.yaw;
  const pitch = lastResponse?.result?.signals?.pitch;

  useEffect(() => {
    return () => {
      stopRealtime();
    };
  }, []);

  async function startRealtime() {
    try {
      setCameraError("");
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 },
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await new Promise<void>((resolve, reject) => {
          const video = videoRef.current;
          if (!video) {
            reject(new Error("视频组件未初始化"));
            return;
          }

          const onLoadedMetadata = async () => {
            video.removeEventListener("loadedmetadata", onLoadedMetadata);
            try {
              await video.play();
              resolve();
            } catch (error) {
              reject(error);
            }
          };

          if (video.readyState >= 1 && video.videoWidth > 0 && video.videoHeight > 0) {
            void video.play().then(() => resolve()).catch(reject);
            return;
          }

          video.addEventListener("loadedmetadata", onLoadedMetadata, { once: true });
        });
      }
      connect();
      setRunning(true);

      let frameId = 0;
      timerRef.current = window.setInterval(() => {
        if (!videoRef.current) {
          return;
        }
        if (videoRef.current.videoWidth <= 0 || videoRef.current.videoHeight <= 0) {
          return;
        }
        const image = frameToBase64(videoRef.current);
        if (!image) {
          return;
        }
        sendFrame(frameId++, image);
      }, 500);
    } catch (err) {
      setCameraError(err instanceof Error ? err.message : "摄像头启动失败");
    }
  }

  function stopRealtime() {
    if (timerRef.current !== null) {
      window.clearInterval(timerRef.current);
      timerRef.current = null;
    }
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    disconnect();
    setRunning(false);
  }

  return (
    <section className="panel realtime-panel">
      <div className="panel-header">
        <h3>实时检测</h3>
        <div className="realtime-actions">
          <button className="action-button" disabled={running} onClick={startRealtime}>
            开始
          </button>
          <button className="ghost-button" disabled={!running} onClick={stopRealtime}>
            停止
          </button>
        </div>
      </div>
      <div className="realtime-layout">
        <div className="video-box">
          <video ref={videoRef} muted playsInline />
          <div className="status-bar">
            <span>{connected ? "WebSocket 已连接" : "WebSocket 未连接"}</span>
            <span>{running ? "摄像头运行中" : "摄像头未启动"}</span>
          </div>
          <div className="debug-panel">
            <div className="debug-chip-group">
              <span className={`debug-chip ${hasRealtimeResult ? "active" : "idle"}`}>
                {hasRealtimeResult ? "已收到实时结果" : "等待实时结果"}
              </span>
              <span className={`debug-chip ${faceDetected ? "active" : "warn"}`}>
                {faceDetected ? "已检测到人脸" : "当前未检测到人脸"}
              </span>
              <span className="debug-chip neutral">当前风险等级：{riskLevelText}</span>
            </div>
            {!faceDetected && running ? (
              <div className="debug-hint">
                请将脸部置于画面中央，距离镜头约 30 到 50 厘米，保持正面朝向并确保光线充足。
              </div>
            ) : null}
            {hasRealtimeResult ? (
              <div className="realtime-metrics-grid">
                <div className="realtime-metric-item">
                  <span>EAR</span>
                  <strong>{typeof ear === "number" ? ear.toFixed(3) : "--"}</strong>
                </div>
                <div className="realtime-metric-item">
                  <span>MAR</span>
                  <strong>{typeof mar === "number" ? mar.toFixed(3) : "--"}</strong>
                </div>
                <div className="realtime-metric-item">
                  <span>偏航角</span>
                  <strong>{typeof yaw === "number" ? yaw.toFixed(1) : "--"}</strong>
                </div>
                <div className="realtime-metric-item">
                  <span>俯仰角</span>
                  <strong>{typeof pitch === "number" ? pitch.toFixed(1) : "--"}</strong>
                </div>
              </div>
            ) : null}
          </div>
          {cameraError ? <div className="inline-error">{cameraError}</div> : null}
          {error ? <div className="inline-error">{error}</div> : null}
        </div>
        <ResultCard title="实时推理结果" result={lastResponse?.result ?? null} />
      </div>
    </section>
  );
}
