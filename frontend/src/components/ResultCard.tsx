import type { UnifiedInferenceResult } from "../types/api";

interface ResultCardProps {
  title: string;
  result: UnifiedInferenceResult | null;
}

const riskLevelText: Record<string, string> = {
  normal: "正常",
  mild: "轻度",
  moderate: "中度",
  severe: "重度",
};

const signalLabelMap: Record<string, string> = {
  face_detected: "检测到人脸",
  ear: "眼部闭合指标 EAR",
  mar: "嘴部张开指标 MAR",
  yaw: "头部偏航角",
  pitch: "头部俯仰角",
  roll: "头部翻滚角",
  eye_closed_rule: "闭眼规则触发",
  yawn_rule: "打哈欠规则触发",
  head_turned_rule: "偏头规则触发",
  head_down_rule: "低头规则触发",
  rule_fatigue_score: "规则疲劳分",
  rule_distraction_score: "规则分心分",
  fusion_notes: "融合说明",
};

const modelProbLabelMap: Record<string, string> = {
  normal: "正常",
  eye_closed: "闭眼",
  yawn: "打哈欠",
  distracted: "分心",
};

const alertLabelMap: Record<string, string> = {
  fatigue_eye_closure: "闭眼疲劳告警",
  moderate_fatigue: "中度疲劳",
  severe_fatigue: "重度疲劳",
  distraction_detected: "检测到分心",
  moderate_distraction: "中度分心",
  severe_distraction: "重度分心",
  model_supported_eye_closed: "模型支持闭眼判断",
  model_supported_yawn: "模型支持哈欠判断",
  model_supported_distraction: "模型支持分心判断",
};

function formatLabel(key: string, mapping: Record<string, string>): string {
  return mapping[key] ?? key;
}

function renderSignalEntries(signals: Record<string, unknown>) {
  return Object.entries(signals).slice(0, 8).map(([key, value]) => (
    <div key={key} className="kv-row">
      <span>{formatLabel(key, signalLabelMap)}</span>
      <span>{String(value)}</span>
    </div>
  ));
}

function renderProbEntries(modelProbs: Record<string, number>) {
  return Object.entries(modelProbs).map(([key, value]) => (
    <div key={key} className="kv-row">
      <span>{formatLabel(key, modelProbLabelMap)}</span>
      <span>{value.toFixed(4)}</span>
    </div>
  ));
}

function renderListEntries(values: string[], mapper?: Record<string, string>) {
  if (values.length === 0) {
    return <span className="alert-chip quiet">暂无</span>;
  }

  return values.map((value) => (
    <span key={value} className="alert-chip quiet">
      {mapper ? formatLabel(value, mapper) : value}
    </span>
  ));
}

export default function ResultCard({ title, result }: ResultCardProps) {
  return (
    <section className="panel">
      <div className="panel-header">
        <h3>{title}</h3>
      </div>
      {!result ? (
        <div className="empty-state">暂无结果</div>
      ) : (
        <div className="result-layout">
          <div className="metric-grid">
            <div className="metric-tile">
              <span>规则法风险</span>
              <strong>{formatLabel(result.rule_result.risk_level, riskLevelText)}</strong>
            </div>
            <div className="metric-tile">
              <span>模型分类</span>
              <strong>
                {result.model_result.predicted_label
                  ? formatLabel(result.model_result.predicted_label, modelProbLabelMap)
                  : "未输出"}
              </strong>
            </div>
            <div className="metric-tile">
              <span>融合后风险</span>
              <strong>{formatLabel(result.fusion_result.risk_level, riskLevelText)}</strong>
            </div>
            <div className="metric-tile">
              <span>模型置信度</span>
              <strong>{(result.model_result.predicted_confidence * 100).toFixed(1)}%</strong>
            </div>
          </div>

          <div className="detail-columns">
            <div className="detail-card">
              <h4>规则法输出</h4>
              <div className="kv-row">
                <span>疲劳分</span>
                <span>{result.rule_result.fatigue_score.toFixed(2)}</span>
              </div>
              <div className="kv-row">
                <span>分心分</span>
                <span>{result.rule_result.distraction_score.toFixed(2)}</span>
              </div>
              <div className="kv-row">
                <span>风险总分</span>
                <span>{result.rule_result.risk_score.toFixed(2)}</span>
              </div>
              <div className="alert-row compact-alert-row">
                {renderListEntries(result.rule_result.status_labels)}
              </div>
            </div>
            <div className="detail-card">
              <h4>模型法输出</h4>
              <div className="kv-row">
                <span>是否启用模型</span>
                <span>{result.model_result.enabled ? "是" : "否"}</span>
              </div>
              <div className="kv-row">
                <span>是否使用人脸 ROI</span>
                <span>{result.model_result.face_roi_used ? "是" : "否"}</span>
              </div>
              <div className="kv-row">
                <span>预测标签</span>
                <span>
                  {result.model_result.predicted_label
                    ? formatLabel(result.model_result.predicted_label, modelProbLabelMap)
                    : "无"}
                </span>
              </div>
              <div className="kv-row">
                <span>预测置信度</span>
                <span>{(result.model_result.predicted_confidence * 100).toFixed(1)}%</span>
              </div>
              <div className="detail-card-inline">
                {renderProbEntries(result.model_probs)}
              </div>
            </div>
          </div>

          <div className="detail-columns">
            <div className="detail-card">
              <h4>融合后结果</h4>
              <div className="kv-row">
                <span>疲劳分</span>
                <span>{result.fusion_result.fatigue_score.toFixed(2)}</span>
              </div>
              <div className="kv-row">
                <span>分心分</span>
                <span>{result.fusion_result.distraction_score.toFixed(2)}</span>
              </div>
              <div className="kv-row">
                <span>风险等级</span>
                <span>{formatLabel(result.fusion_result.risk_level, riskLevelText)}</span>
              </div>
              <div className="alert-row compact-alert-row">
                {result.fusion_result.alerts.length === 0 ? (
                  <span className="alert-chip quiet">当前无告警</span>
                ) : (
                  result.fusion_result.alerts.map((alert) => (
                    <span key={alert} className="alert-chip">
                      {formatLabel(alert, alertLabelMap)}
                    </span>
                  ))
                )}
              </div>
            </div>
            <div className="detail-card">
              <h4>规则底层信号</h4>
              {renderSignalEntries(result.signals)}
            </div>
          </div>

          <div className="detail-card">
            <h4>规则原因</h4>
            <div className="alert-row compact-alert-row">
              {renderListEntries(result.rule_result.reasons)}
            </div>
          </div>
        </div>
      )}
    </section>
  );
}
