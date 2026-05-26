import type { TrainingMetricsResponse } from "../types/api";

interface TrainingMetricsPanelProps {
  data: TrainingMetricsResponse | null;
}

const classLabelMap: Record<string, string> = {
  normal: "正常",
  eye_closed: "闭眼",
  yawn: "打哈欠",
  distracted: "分心",
};

function formatPercent(value: number): string {
  return `${(value * 100).toFixed(2)}%`;
}

export default function TrainingMetricsPanel({
  data,
}: TrainingMetricsPanelProps) {
  if (!data) {
    return (
      <section className="panel">
        <div className="panel-header">
          <h3>训练成果</h3>
        </div>
        <div className="empty-state">训练指标加载中</div>
      </section>
    );
  }

  return (
    <section className="panel">
      <div className="panel-header">
        <h3>训练成果</h3>
      </div>

      <div className="metric-grid">
        <div className="metric-tile">
          <span>模型</span>
          <strong>{data.model_name}</strong>
        </div>
        <div className="metric-tile">
          <span>最佳轮次</span>
          <strong>{data.best_epoch}</strong>
        </div>
        <div className="metric-tile">
          <span>准确率</span>
          <strong>{formatPercent(data.metrics.accuracy)}</strong>
        </div>
        <div className="metric-tile">
          <span>宏平均 F1</span>
          <strong>{formatPercent(data.metrics.f1_macro)}</strong>
        </div>
        <div className="metric-tile">
          <span>精确率</span>
          <strong>{formatPercent(data.metrics.precision_macro)}</strong>
        </div>
        <div className="metric-tile">
          <span>召回率</span>
          <strong>{formatPercent(data.metrics.recall_macro)}</strong>
        </div>
      </div>

      <div className="detail-columns">
        <div className="detail-card">
          <h4>分类别指标</h4>
          {data.class_names.map((className, index) => (
            <div key={className} className="kv-block">
              <div className="kv-row">
                <span>{classLabelMap[className] ?? className}</span>
                <span>F1 {formatPercent(data.metrics.f1_per_class[index] ?? 0)}</span>
              </div>
              <div className="kv-row">
                <span>精确率</span>
                <span>{formatPercent(data.metrics.precision_per_class[index] ?? 0)}</span>
              </div>
              <div className="kv-row">
                <span>召回率</span>
                <span>{formatPercent(data.metrics.recall_per_class[index] ?? 0)}</span>
              </div>
            </div>
          ))}
        </div>

        <div className="detail-card">
          <h4>混淆矩阵</h4>
          <div className="confusion-grid">
            <div className="confusion-row confusion-header">
              <span></span>
              {data.class_names.map((label) => (
                <span key={label}>{classLabelMap[label] ?? label}</span>
              ))}
            </div>
            {data.metrics.confusion_matrix.map((row, rowIndex) => (
              <div key={data.class_names[rowIndex]} className="confusion-row">
                <span>{classLabelMap[data.class_names[rowIndex]] ?? data.class_names[rowIndex]}</span>
                {row.map((value, columnIndex) => (
                  <span
                    key={`${rowIndex}-${columnIndex}`}
                    className={rowIndex === columnIndex ? "diag-cell" : ""}
                  >
                    {value}
                  </span>
                ))}
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="detail-card">
        <h4>训练曲线快照</h4>
        <div className="curve-list">
          {data.train_curve.slice(-6).map((point) => (
            <div key={point.epoch} className="kv-row">
              <span>第 {point.epoch} 轮</span>
              <span>
                验证损失 {point.val_loss.toFixed(4)} · 验证准确率{" "}
                {formatPercent(point.val_accuracy)} · 验证 F1 {formatPercent(point.val_f1)}
              </span>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
