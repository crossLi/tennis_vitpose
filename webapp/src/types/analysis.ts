export type AnalysisStage =
  | "queued"
  | "classifying"
  | "rejected"
  | "generating_pose_video"
  | "extracting_features"
  | "summarizing_features"
  | "analyzing_with_kimi"
  | "completed"
  | "failed";

export type ArtifactUrls = Partial<Record<ArtifactKind, string>>;

export type ArtifactKind =
  | "original_video"
  | "original_video_mp4"
  | "pose_video"
  | "pose_sequence"
  | "features"
  | "summary"
  | "kimi_result";

export type AnalysisArtifacts = Partial<Record<ArtifactKind, string>> & {
  urls?: ArtifactUrls;
};

export type AnalysisResult = {
  kimiAnalysis?: string;
  kimiStructured?: CoachAnalysisPayload | null;
  summaryText?: string;
  summaryStructured?: SummaryPayload | null;
  classificationReason?: string;
};

export type AnalysisTask = {
  taskId: string;
  stage: AnalysisStage;
  message: string;
  isTennisVideo?: boolean;
  rejectionReason?: string;
  progress: number;
  createdAt: string;
  updatedAt: string;
  artifacts: AnalysisArtifacts;
  result: AnalysisResult;
};

export type SummaryPayload = {
  video_summary?: Record<string, unknown>;
  stance_type_distribution?: Record<string, number>;
  dominant_side?: Record<string, unknown>;
  action_phase_segments?: Array<Record<string, unknown>>;
  feature_stats?: Record<string, unknown>;
  coordination_metrics?: Record<string, unknown>;
  head_stability_summary?: Record<string, unknown>;
  symmetry_summary?: Record<string, unknown>;
  llm_brief?: string;
};

export type CoachAnalysisPayload = {
  overview?: {
    score?: number;
    coach_title?: string;
    coach_advice?: string;
  };
  highlights?: {
    section_title?: string;
    section_label?: string;
    headline?: string;
    timestamp_sec?: number;
    phase?: string;
    bullets?: string[];
  };
  improvements?: {
    section_title?: string;
    section_label?: string;
    segments?: CoachImprovementSegment[];
  };
  skills_status?: {
    section_title?: string;
    items?: CoachSkillStatusItem[];
  };
  training_plan_cta?: string;
  confidence?: {
    overall_confidence?: number;
    limitations?: string[];
  };
};

export type CoachImprovementSegment = {
  timestamp_sec?: number;
  phase?: string;
  headline?: string;
  frame_caption?: string;
  checkpoints?: CoachCheckpoint[];
};

export type CoachCheckpoint = {
  name?: string;
  status?: "good" | "bad";
  detail?: string;
};

export type CoachSkillStatusItem = {
  name?: string;
  status?: "pass" | "needs_work";
  detail?: string;
};
