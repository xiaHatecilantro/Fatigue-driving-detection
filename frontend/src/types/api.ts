export type RiskLevel = "normal" | "mild" | "moderate" | "severe";

export interface RuleInferenceResult {
  fatigue_score: number;
  distraction_score: number;
  risk_score: number;
  risk_level: RiskLevel;
  status_labels: string[];
  reasons: string[];
  alarm_on: boolean;
}

export interface ModelInferenceSummary {
  enabled: boolean;
  face_roi_used: boolean;
  predicted_label?: string | null;
  predicted_confidence: number;
}

export interface FusionSummary {
  fatigue_score: number;
  distraction_score: number;
  risk_level: RiskLevel;
  alerts: string[];
}

export interface UnifiedInferenceResult {
  fatigue_score: number;
  distraction_score: number;
  risk_level: RiskLevel;
  signals: Record<string, unknown>;
  model_probs: Record<string, number>;
  alerts: string[];
  timestamp: string;
  rule_result: RuleInferenceResult;
  model_result: ModelInferenceSummary;
  fusion_result: FusionSummary;
}

export interface ImageInferenceResponse {
  status: string;
  message: string;
  result: UnifiedInferenceResult;
  visualization_path?: string | null;
}

export interface VideoInferenceSummary {
  total_frames: number;
  processed_frames: number;
  failed_frames: number;
  fps: number;
  average_risk_score: number;
  peak_risk_score: number;
  peak_risk_level: RiskLevel;
  event_count: number;
  saved_visualization: boolean;
}

export interface VideoInferenceResponse {
  status: string;
  message: string;
  summary: VideoInferenceSummary;
  results: UnifiedInferenceResult[];
  events: Record<string, unknown>[];
  visualization_path?: string | null;
}

export interface RealtimeSocketResponse {
  status: string;
  frame_id?: number;
  message?: string;
  result?: UnifiedInferenceResult;
}

export interface TrainingMetrics {
  accuracy: number;
  precision_macro: number;
  recall_macro: number;
  f1_macro: number;
  precision_per_class: number[];
  recall_per_class: number[];
  f1_per_class: number[];
  confusion_matrix: number[][];
}

export interface TrainingCurvePoint {
  epoch: number;
  train_loss: number;
  val_loss: number;
  val_accuracy: number;
  val_f1: number;
}

export interface TrainingMetricsResponse {
  model_name: string;
  checkpoint: string;
  best_epoch: number;
  class_names: string[];
  metrics: TrainingMetrics;
  train_curve: TrainingCurvePoint[];
}
