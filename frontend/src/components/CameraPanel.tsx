import { useCallback, useRef, useState } from "react";
import { inferImage } from "../services/api";
import type { ImageInferenceResponse } from "../types/api";

export default function CameraPanel() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const sendCanvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const [cameraReady, setCameraReady] = useState(false);
  const [captures, setCaptures] = useState<ImageInferenceResponse[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);
  const [judgments, setJudgments] = useState<Record<string, "correct" | "incorrect" | null>>({});

  /* ---------- camera ---------- */

  const startCamera = useCallback(async () => {
    try {
      setError("");
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
      });
      streamRef.current = stream;
      if (videoRef.current) videoRef.current.srcObject = stream;
      setCameraReady(true);
    } catch {
      setError("无法打开摄像头，请确认已授权相机权限。");
    }
  }, []);

  const stopCamera = useCallback(() => {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setCameraReady(false);
    setError("");
  }, []);

  /* ---------- capture ---------- */

  const capture = useCallback(async () => {
    const video = videoRef.current;
    const canvas = sendCanvasRef.current;
    if (!video || !canvas) return;
    setBusy(true);
    setError("");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d")?.drawImage(video, 0, 0);
    try {
      const blob = await new Promise<Blob>((r, e) =>
        canvas.toBlob((b) => (b ? r(b) : e(new Error("截图失败"))), "image/jpeg", 0.92));
      const file = new File([blob], `capture_${Date.now()}.jpg`, { type: "image/jpeg" });
      const result = await inferImage(file, true);
      setCaptures((prev) => [result, ...prev]);
    } catch (e) {
      setError(e instanceof Error ? e.message : "推理失败");
    } finally { setBusy(false); }
  }, []);

  const selected = selectedIndex !== null ? captures[selectedIndex] : null;

  return (
    <div className="camera-shell">
      {/* ===== 左栏 ===== */}
      <div className="camera-left">
        <div className="camera-video-box">
          <video ref={videoRef} autoPlay playsInline muted className="camera-video" />
          <canvas ref={sendCanvasRef} style={{ display: "none" }} />

          {!cameraReady && !error && (
            <div className="camera-overlay">
              <p>摄像头未开启</p>
              <button className="action-button" onClick={startCamera}>打开摄像头</button>
            </div>
          )}
          {error && (
            <div className="camera-overlay camera-error">
              <p>{error}</p>
              <button className="action-button" onClick={startCamera}>重新尝试</button>
            </div>
          )}
        </div>

        <div className="camera-actions">
          {cameraReady
            ? <button className="ghost-button" onClick={stopCamera}>关闭摄像头</button>
            : <button className="action-button" onClick={startCamera}>打开摄像头</button>
          }
          <button className="action-button capture-btn" disabled={!cameraReady || busy} onClick={capture}>
            {busy ? "推理中..." : "拍照"}
          </button>
        </div>
      </div>

      {/* ===== 右栏 ===== */}
      <div className="camera-right">
        <div className="capture-list-header">
          <h3 className="capture-list-title">已拍照 <span className="capture-count">{captures.length}</span></h3>
          {captures.length > 0 && (
            <button className="clear-btn" onClick={() => { setCaptures([]); setSelectedIndex(null); }}>清空</button>
          )}
        </div>
        {captures.length === 0 ? (
          <div className="empty-state">拍照后将在此处展示识别结果</div>
        ) : (
          <div className="capture-list">
            {captures.map((cap, i) => {
              const imgSrc = cap.visualization_path ? `http://127.0.0.1:8000/${cap.visualization_path}` : null;
              return (
                <div key={`${cap.result.timestamp}-${i}`}
                  className={`capture-card ${selectedIndex === i ? "capture-card--active" : ""}`}
                  onClick={() => setSelectedIndex(i)}>
                  <div className="capture-thumb-wrap">
                    {imgSrc
                      ? <img src={imgSrc} alt="" className="capture-thumb" />
                      : <div className="capture-thumb capture-thumb--empty">无标注图</div>}
                    {judgments[cap.visualization_path ?? ""] && (
                      <span className={`thumb-judge thumb-judge--${judgments[cap.visualization_path ?? ""]}`}>
                        {judgments[cap.visualization_path ?? ""] === "correct" ? "✓" : "✗"}
                      </span>
                    )}
                  </div>
                  <div className="capture-meta">
                    <span className={`risk-badge risk-${cap.result.fusion_result.risk_level}`}>
                      {riskLabel(cap.result.fusion_result.risk_level)}
                    </span>
                    <span className="capture-time">{formatTime(cap.result.timestamp)}</span>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* ===== 弹窗 ===== */}
      {selected && (
        <div className="modal-backdrop" onClick={() => setSelectedIndex(null)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <button className="modal-close" onClick={() => setSelectedIndex(null)}>✕</button>
            {selected.visualization_path
              ? <img src={`http://127.0.0.1:8000/${selected.visualization_path}`} alt="" className="modal-image" />
              : <div className="modal-image modal-image--empty">无标注图</div>}

            <div className="modal-judge-bar">
              <span className="judge-label">模型判断：</span>
              <span className="judge-class">
                {labelMap[selected.result.model_result.predicted_label ?? ""] ?? selected.result.model_result.predicted_label ?? "未检出"}
                {" "}{(selected.result.model_result.predicted_confidence * 100).toFixed(1)}%
              </span>
              <div className="judge-buttons">
                <button className={`judge-btn judge-correct ${judgments[selected.visualization_path ?? ""] === "correct" ? "judge-btn--active" : ""}`}
                  onClick={() => setJudgments((p) => ({ ...p, [selected.visualization_path ?? ""]: "correct" }))}>
                  ✓ 正确
                </button>
                <button className={`judge-btn judge-incorrect ${judgments[selected.visualization_path ?? ""] === "incorrect" ? "judge-btn--active" : ""}`}
                  onClick={() => setJudgments((p) => ({ ...p, [selected.visualization_path ?? ""]: "incorrect" }))}>
                  ✗ 错误
                </button>
              </div>
              {judgments[selected.visualization_path ?? ""] && (
                <span className="judge-done">{judgments[selected.visualization_path ?? ""] === "correct" ? "✓ 已标记为正确" : "✗ 已标记为错误"}</span>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

const labelMap: Record<string, string> = { normal: "正常", eye_closed: "闭眼", yawn: "打哈欠", distracted: "分心" };

function formatTime(ts: string) {
  try { return new Date(ts).toLocaleTimeString("zh-CN", { hour12: false }); } catch { return ts; }
}

function riskLabel(level: string): string {
  return ({ normal: "正常", mild: "轻度", moderate: "中度", severe: "重度" } as Record<string, string>)[level] ?? level;
}
