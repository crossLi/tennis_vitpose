import { useEffect, useMemo, useRef, useState } from "react";
import { AlertTriangle, CheckCircle2, Play, XCircle } from "lucide-react";

import { fetchTextArtifact, resolveArtifactUrl, resolveTaskFrameUrl } from "@/lib/api";
import type {
  AnalysisTask,
  CoachAnalysisPayload,
  CoachCheckpoint,
  CoachImprovementSegment,
  CoachSkillStatusItem,
  SummaryPayload,
} from "@/types/analysis";

type ResultPanelProps = {
  task: AnalysisTask;
};

type ScoreLevel = "excellent" | "good" | "average";

type PhaseKey = "backswing" | "finish";

const PHASE_LABEL: Record<PhaseKey, string> = {
  backswing: "蓄力引拍",
  finish: "挥拍收拍",
};

const SKILL_WHITELIST = ["腿部转换", "挥拍完整度", "脚步", "下肢驱动"];

function resolveScoreLevel(score: number): ScoreLevel {
  if (score >= 90) {
    return "excellent";
  }
  if (score >= 60) {
    return "good";
  }
  return "average";
}

const SCORE_LABEL: Record<ScoreLevel, string> = {
  excellent: "优秀",
  good: "良好",
  average: "一般",
};

function parseJsonPayload<T>(text?: string): T | null {
  if (!text) {
    return null;
  }
  try {
    return JSON.parse(text) as T;
  } catch {
    return null;
  }
}

function formatTimestamp(value?: number): string {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return "--";
  }
  return `${value.toFixed(2)}s`;
}

function frameTimes(timestampSec?: number): number[] {
  if (typeof timestampSec !== "number" || Number.isNaN(timestampSec)) {
    return [];
  }
  return [timestampSec - 0.08, timestampSec, timestampSec + 0.08].map((value) =>
    Math.max(0, Number(value.toFixed(3))),
  );
}

function detectPhaseKey(phase?: string, headline?: string): PhaseKey {
  const text = `${phase ?? ""} ${headline ?? ""}`.toLowerCase();
  if (text.includes("蓄力") || text.includes("引拍") || text.includes("准备") || text.includes("backswing")) {
    return "backswing";
  }
  if (text.includes("收拍") || text.includes("随挥") || text.includes("挥拍") || text.includes("击球") || text.includes("follow") || text.includes("finish")) {
    return "finish";
  }
  return "backswing";
}

function normalizeSkillName(name: string): string {
  if (/下肢|驱动/.test(name)) {
    return "下肢驱动";
  }
  if (/腿部|蹬转|转换/.test(name)) {
    return "腿部转换";
  }
  if (/挥拍|完整|随挥/.test(name)) {
    return "挥拍完整度";
  }
  if (/脚步|步法|站位|分腿|垫步/.test(name)) {
    return "脚步";
  }
  return name;
}

type MobileVideoPlayerProps = {
  src: string;
  kind: "original_video" | "pose_video";
  taskId: string;
  posterTimeSec?: number;
};

function MobileVideoPlayer({ src, kind, taskId, posterTimeSec = 0.1 }: MobileVideoPlayerProps) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [hasError, setHasError] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const posterUrl = useMemo(
    () => resolveTaskFrameUrl(taskId, posterTimeSec, kind),
    [taskId, kind, posterTimeSec],
  );

  useEffect(() => {
    const video = videoRef.current;
    if (!video) {
      return;
    }
    setIsLoading(true);
    setHasError(false);
    setIsPlaying(false);
    video.setAttribute("webkit-playsinline", "true");
    video.setAttribute("playsinline", "true");
    video.setAttribute("x5-playsinline", "true");
    video.setAttribute("x5-video-player-type", "h5-page");
    video.setAttribute("x5-video-player-fullscreen", "false");
    video.setAttribute("x-webkit-airplay", "allow");
    try {
      void video.load();
      if (video.paused) {
        const playResult = video.play();
        if (playResult && typeof playResult.catch === "function") {
          playResult.catch(() => undefined);
        }
      }
    } catch {
      // ignore
    }
  }, [src]);

  const handleClick = () => {
    const video = videoRef.current;
    if (!video) {
      return;
    }
    if (video.paused || video.ended) {
      const playPromise = video.play();
      if (playPromise && typeof playPromise.catch === "function") {
        playPromise.catch(() => undefined);
      }
    } else {
      video.pause();
    }
  };

  return (
    <div className="relative h-44 w-full overflow-hidden rounded-2xl border border-[#ECECF1] bg-black md:h-72 md:rounded-[24px]">
      <video
        ref={videoRef}
        src={src}
        poster={posterUrl}
        controls
        controlsList="nodownload"
        playsInline
        muted
        autoPlay
        loop={false}
        preload="auto"
        onClick={handleClick}
        onPlay={() => setIsPlaying(true)}
        onPlaying={() => {
          setIsPlaying(true);
          setIsLoading(false);
        }}
        onPause={() => setIsPlaying(false)}
        onWaiting={() => setIsLoading(true)}
        onCanPlay={() => setIsLoading(false)}
        onLoadedData={() => setIsLoading(false)}
        onError={() => {
          setHasError(true);
          setIsLoading(false);
        }}
        className={`h-full w-full object-contain ${hasError ? "opacity-50" : ""}`}
      />
      {isLoading && !hasError ? (
        <div className="pointer-events-none absolute inset-0 flex items-center justify-center bg-black/40">
          <div className="flex flex-col items-center gap-2">
            <div className="size-7 animate-spin rounded-full border-2 border-white/20 border-t-white md:size-9" />
            <span className="text-xs text-white/80">视频加载中…</span>
          </div>
        </div>
      ) : null}
      {hasError ? (
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 bg-black/70 px-4 text-center">
          <XCircle className="size-6 text-rose-400 md:size-8" />
          <p className="text-xs text-white/90 md:text-sm">视频加载失败</p>
          <p className="text-[11px] text-white/60 md:text-xs">点击播放按钮重试，或刷新页面</p>
        </div>
      ) : null}
      {!isPlaying && !isLoading && !hasError ? (
        <div className="pointer-events-none absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2">
          <div className="flex size-12 items-center justify-center rounded-full bg-black/55 backdrop-blur-sm ring-1 ring-white/20 md:size-16">
            <Play className="size-5 fill-white text-white md:size-7" />
          </div>
        </div>
      ) : null}
    </div>
  );
}

function ImprovementFrames({ taskId, segment }: { taskId: string; segment: CoachImprovementSegment }) {
  const times = frameTimes(segment.timestamp_sec);
  if (!times.length) {
    return null;
  }

  return (
    <div className="grid grid-cols-3 gap-2 md:gap-3">
      {times.map((time) => (
        <div
          key={`${segment.headline}-${time}`}
          className="overflow-hidden rounded-xl border border-[#ECECF1] bg-white md:rounded-[24px]"
        >
          <img
            src={resolveTaskFrameUrl(taskId, time, "original_video")}
            alt={`动作帧 ${formatTimestamp(time)}`}
            loading="lazy"
            className="h-24 w-full object-cover md:h-44"
          />
          <div className="border-t border-[#ECECF1] px-2 py-1.5 text-center text-[10px] tracking-[0.12em] text-[#666666] md:px-3 md:py-2 md:text-xs md:tracking-[0.16em]">
            {formatTimestamp(time)}
          </div>
        </div>
      ))}
    </div>
  );
}

type PhasedSegments = Record<PhaseKey, CoachImprovementSegment[]>;

function splitSegmentsByPhase(segments: CoachImprovementSegment[] | undefined): PhasedSegments {
  const result: PhasedSegments = { backswing: [], finish: [] };
  for (const segment of segments ?? []) {
    const key = detectPhaseKey(segment.phase, segment.headline);
    result[key].push(segment);
  }
  return result;
}

function CheckpointIcon({ status }: { status?: CoachCheckpoint["status"] }) {
  return status === "good" ? (
    <CheckCircle2 className="size-3.5 md:size-4" />
  ) : (
    <XCircle className="size-3.5 md:size-4" />
  );
}

export function ResultPanel({ task }: ResultPanelProps) {
  const [showVideos, setShowVideos] = useState(false);
  const [activePhase, setActivePhase] = useState<PhaseKey>("backswing");
  const [analysisText, setAnalysisText] = useState(task.result.kimiAnalysis || "");
  const [summaryTextRaw, setSummaryTextRaw] = useState(task.result.summaryText || "");

  const originalVideoUrl = resolveArtifactUrl(task, "original_video");
  const poseVideoUrl = resolveArtifactUrl(task, "pose_video");
  const summaryUrl = resolveArtifactUrl(task, "summary");
  const kimiUrl = resolveArtifactUrl(task, "kimi_result");

  const summary = useMemo(
    () => task.result.summaryStructured ?? parseJsonPayload<SummaryPayload>(summaryTextRaw),
    [summaryTextRaw, task.result.summaryStructured],
  );
  const coachAnalysis = useMemo(
    () => task.result.kimiStructured ?? parseJsonPayload<CoachAnalysisPayload>(analysisText),
    [analysisText, task.result.kimiStructured],
  );
  void summary;

  const score = Math.max(0, Math.min(100, Math.round(coachAnalysis?.overview?.score ?? 0)));
  const scoreLevel = resolveScoreLevel(score);
  const coreAdvice = coachAnalysis?.overview?.coach_advice || "模型尚未生成核心建议。";

  const phasedSegments = useMemo(
    () => splitSegmentsByPhase(coachAnalysis?.improvements?.segments),
    [coachAnalysis],
  );

  const skillItems: CoachSkillStatusItem[] = useMemo(() => {
    const items = coachAnalysis?.skills_status?.items ?? [];
    const normalized: CoachSkillStatusItem[] = [];
    const used = new Set<string>();
    for (const item of items) {
      const canonicalName = normalizeSkillName(item.name ?? "");
      if (!SKILL_WHITELIST.includes(canonicalName) || used.has(canonicalName)) {
        continue;
      }
      used.add(canonicalName);
      normalized.push({
        ...item,
        name: canonicalName,
      });
    }
    for (const want of SKILL_WHITELIST) {
      if (!used.has(want)) {
        normalized.push({ name: want, status: "needs_work", detail: "当前数据不足，暂未评估。" });
      }
    }
    return normalized.slice(0, 4);
  }, [coachAnalysis]);

  useEffect(() => {
    let cancelled = false;
    async function loadTexts() {
      try {
        const jobs: Promise<void>[] = [];
        if (summaryUrl && !summaryTextRaw) {
          jobs.push(
            fetchTextArtifact(summaryUrl).then((text) => {
              if (!cancelled) {
                setSummaryTextRaw(text);
              }
            }),
          );
        }
        if (kimiUrl && !analysisText) {
          jobs.push(
            fetchTextArtifact(kimiUrl).then((text) => {
              if (!cancelled) {
                setAnalysisText(text);
              }
            }),
          );
        }
        await Promise.all(jobs);
      } catch {
        // ignore
      }
    }
    if (["completed", "rejected", "failed"].includes(task.stage)) {
      void loadTexts();
    }
    return () => {
      cancelled = true;
    };
  }, [analysisText, kimiUrl, summaryTextRaw, summaryUrl, task.stage]);

  if (task.stage === "rejected") {
    return (
      <section className="report-card border-[#FDE68A] bg-[#FFFBEB] text-[#78350F]">
        <div className="mb-3.5 flex items-center gap-2.5 md:mb-4 md:gap-3">
          <AlertTriangle className="size-5 md:size-6" />
          <h2 className="text-lg font-semibold md:text-xl">判断不是网球视频，无法进行分析</h2>
        </div>
        <p className="max-w-3xl text-xs leading-6 text-[#78350F]/90 md:text-sm md:leading-7">
          {task.result.classificationReason ||
            "模型没有识别到“真人正在击打网球”的明确场景，因此系统终止了后续姿态和专项分析。"}
        </p>
      </section>
    );
  }

  const overviewClass =
    scoreLevel === "excellent"
      ? "status-excellent"
      : scoreLevel === "good"
        ? "status-good"
        : "status-average";

  return (
    <section className="grid gap-5 md:gap-7">
      <section
        className="rounded-2xl border border-[#ECECF1] bg-white p-5 shadow-sm md:rounded-[32px] md:p-8"
        style={{
          background:
            "linear-gradient(135deg, #ffffff 0%, #f5f3ff 60%, #fbf7ff 100%)",
        }}
      >
        <div className="flex flex-col gap-5 md:gap-7 md:[flex-direction:row] md:items-stretch md:justify-between">
          <div className="flex flex-col items-start gap-3 md:gap-4">
            <span
              className={`rounded-full px-3 py-1.5 text-xs font-medium md:px-4 md:py-1.5 md:text-sm ${overviewClass}`}
            >
              {SCORE_LABEL[scoreLevel]}
            </span>
            <div className="flex items-end gap-2 md:gap-3">
              <span className="text-[64px] leading-none text-[#333333] md:text-[96px]">
                {score}
              </span>
              <span className="pb-2 text-xs text-[#666666] md:pb-3 md:text-base">/100</span>
            </div>
          </div>
          <div className="min-w-0 flex-1 rounded-2xl border border-[#ECECF1] bg-white/80 p-4 md:min-w-[280px] md:rounded-[24px] md:p-6">
            <p className="mb-2 text-xs font-medium text-[#7F68FF] md:mb-3 md:text-sm">核心建议</p>
            <p className="text-sm leading-7 text-[#333333] md:text-base md:leading-8">
              {coreAdvice}
            </p>
          </div>
        </div>
      </section>

      <section className="report-card">
        <div className="mb-4 flex items-center justify-between gap-3 md:mb-6">
          <h3 className="text-lg font-semibold text-[#333333] md:text-xl">关键动作</h3>
        </div>

        <div className="mb-5 flex items-stretch gap-2 rounded-2xl bg-[#F5F3FF] p-1 md:mb-6 md:gap-3 md:rounded-[24px] md:p-1.5">
          {(Object.keys(PHASE_LABEL) as PhaseKey[]).map((key) => {
            const active = activePhase === key;
            return (
              <button
                key={key}
                type="button"
                onClick={() => setActivePhase(key)}
                className={`flex-1 rounded-xl py-2.5 text-sm font-medium transition md:rounded-[20px] md:py-3 md:text-base ${
                  active
                    ? "bg-[#7F68FF] text-white shadow-sm"
                    : "bg-transparent text-[#666666] hover:text-[#333333]"
                }`}
              >
                {PHASE_LABEL[key]}
                <span
                  className={`ml-2 text-[10px] md:text-xs ${
                    active ? "text-white/85" : "text-[#666666]/70"
                  }`}
                >
                  {phasedSegments[key].length}
                </span>
              </button>
            );
          })}
        </div>

        <div className="space-y-4 md:space-y-5">
          {phasedSegments[activePhase].length === 0 ? (
            <div className="rounded-2xl border border-dashed border-[#D1C9FF] bg-white px-4 py-6 text-center text-xs text-[#666666] md:rounded-[24px] md:text-sm">
              暂无{PHASE_LABEL[activePhase]}阶段的关键动作分析。
            </div>
          ) : (
            phasedSegments[activePhase].map((segment, index) => {
              const goodCheckpoints = (segment.checkpoints ?? []).filter((c) => c.status === "good");
              const improveCheckpoints = (segment.checkpoints ?? []).filter(
                (c) => c.status !== "good",
              );
              return (
                <article
                  key={`${segment.headline}-${index}`}
                  className="rounded-2xl border border-[#ECECF1] bg-white p-4 md:rounded-[28px] md:p-6"
                >
                  <div className="mb-4 flex flex-wrap items-start justify-between gap-3 md:mb-5">
                    <div className="min-w-0 flex-1">
                      <div className="text-xs text-[#7F68FF] md:text-sm">
                        {segment.phase || PHASE_LABEL[activePhase]} · {formatTimestamp(segment.timestamp_sec)}
                      </div>
                      <h4 className="mt-1 text-lg font-semibold text-[#333333] md:mt-1.5 md:text-xl">
                        {segment.headline || "该片段存在需要重点打磨的动作问题。"}
                      </h4>
                    </div>
                  </div>

                  <ImprovementFrames taskId={task.taskId} segment={segment} />

                  {goodCheckpoints.length > 0 ? (
                    <div className="mt-4 md:mt-5">
                      <p className="mb-2.5 text-xs font-medium text-[#7F68FF] md:mb-3 md:text-sm">
                        良好 ({goodCheckpoints.length})
                      </p>
                      <div className="grid gap-2.5 md:gap-3 md:grid-cols-2">
                        {goodCheckpoints.map((checkpoint, checkpointIndex) => (
                          <div
                            key={`good-${checkpoint.name}-${checkpointIndex}`}
                            className="rounded-xl border border-[#D1C9FF]/70 bg-[#F5F3FF] p-3 text-[#4B35C7] md:rounded-[22px] md:p-4"
                          >
                            <div className="flex items-center gap-2 text-xs font-medium md:text-sm">
                              <CheckpointIcon status="good" />
                              <span>{checkpoint.name || "检查点"}</span>
                            </div>
                            <p className="mt-1.5 text-xs leading-6 text-[#4B35C7]/90 md:mt-2 md:text-sm md:leading-7">
                              {checkpoint.detail || "模型未生成说明。"}
                            </p>
                          </div>
                        ))}
                      </div>
                    </div>
                  ) : null}

                  {improveCheckpoints.length > 0 ? (
                    <div className="mt-4 md:mt-5">
                      <p className="mb-2.5 text-xs font-medium text-[#725400] md:mb-3 md:text-sm">
                        待提升 ({improveCheckpoints.length})
                      </p>
                      <div className="grid gap-2.5 md:gap-3 md:grid-cols-2">
                        {improveCheckpoints.map((checkpoint, checkpointIndex) => (
                          <div
                            key={`improve-${checkpoint.name}-${checkpointIndex}`}
                            className="rounded-xl border border-[#FFEDAD]/70 bg-[#FFF9E5] p-3 text-[#725400] md:rounded-[22px] md:p-4"
                          >
                            <div className="flex items-center gap-2 text-xs font-medium md:text-sm">
                              <CheckpointIcon status="bad" />
                              <span>{checkpoint.name || "检查点"}</span>
                            </div>
                            <p className="mt-1.5 text-xs leading-6 text-[#725400]/90 md:mt-2 md:text-sm md:leading-7">
                              {checkpoint.detail || "模型未生成说明。"}
                            </p>
                          </div>
                        ))}
                      </div>
                    </div>
                  ) : null}
                </article>
              );
            })
          )}
        </div>
      </section>

      <section className="report-card">
        <div className="mb-4 flex items-center justify-between gap-3 md:mb-6">
          <h3 className="text-lg font-semibold text-[#333333] md:text-xl">技术项</h3>
        </div>
        <div className="grid gap-3 md:gap-4 md:grid-cols-3">
          {skillItems.map((item, index) => {
            const isImprove = item.status === "needs_work";
            return (
              <div
                key={`skill-${item.name}-${index}`}
                className={`rounded-2xl border p-4 md:rounded-[26px] md:p-5 ${
                  isImprove
                    ? "border-[#FFEDAD]/70 bg-[#FFF9E5]"
                    : "border-[#C9EF9B]/70 bg-[#F0FFD9]"
                }`}
              >
                <div className="flex items-center justify-between gap-3">
                  <h4 className="text-base font-semibold text-[#333333] md:text-lg">{item.name}</h4>
                  <span
                    className={`rounded-full px-2.5 py-1 text-[10px] md:px-3 md:text-xs ${
                      isImprove ? "status-improve" : "status-average"
                    }`}
                  >
                    {isImprove ? "待提升" : "一般"}
                  </span>
                </div>
                <p
                  className={`mt-2.5 text-xs leading-6 md:mt-3 md:text-sm md:leading-7 ${
                    isImprove ? "text-[#725400]" : "text-[#294C11]"
                  }`}
                >
                  {item.detail || "模型未生成说明。"}
                </p>
              </div>
            );
          })}
        </div>
      </section>

      <section className="report-card">
        <div className="mb-4 flex items-center justify-between gap-3 md:mb-5">
          <h3 className="text-lg font-semibold text-[#333333] md:text-xl">视频回看</h3>
          <button
            type="button"
            onClick={() => setShowVideos((value) => !value)}
            className="rounded-full border border-[#D1C9FF] bg-[#F5F3FF] px-3.5 py-2 text-xs font-medium text-[#7F68FF] transition hover:bg-[#EDE9FF] md:px-4 md:py-2 md:text-sm"
          >
            {showVideos ? "收起视频" : "加载视频预览"}
          </button>
        </div>
        {showVideos ? (
          <div className="grid gap-4 md:gap-5 md:grid-cols-2">
            <div>
              <p className="mb-2 text-xs text-[#666666] md:mb-3 md:text-sm">原始视频</p>
              {originalVideoUrl ? (
                <MobileVideoPlayer
                  src={originalVideoUrl}
                  kind="original_video"
                  taskId={task.taskId}
                  posterTimeSec={0.08}
                />
              ) : (
                <div className="flex h-44 items-center justify-center rounded-2xl border border-dashed border-[#D1C9FF] bg-[#FAFAFF] text-xs text-[#666666] md:h-72 md:rounded-[24px] md:text-sm">
                  暂无原始视频
                </div>
              )}
            </div>
            <div>
              <p className="mb-2 text-xs text-[#666666] md:mb-3 md:text-sm">姿态可视化</p>
              {poseVideoUrl ? (
                <MobileVideoPlayer
                  src={poseVideoUrl}
                  kind="pose_video"
                  taskId={task.taskId}
                  posterTimeSec={0.08}
                />
              ) : (
                <div className="flex h-44 items-center justify-center rounded-2xl border border-dashed border-[#D1C9FF] bg-[#FAFAFF] text-xs text-[#666666] md:h-72 md:rounded-[24px] md:text-sm">
                  姿态视频生成中…
                </div>
              )}
            </div>
          </div>
        ) : (
          <div className="rounded-xl border border-[#ECECF1] bg-[#FAFAFF] px-4 py-3.5 text-xs text-[#666666] md:rounded-[22px] md:px-4 md:py-4 md:text-sm">
            默认不加载视频播放器，减少首屏渲染压力。需要对照时再点开。
          </div>
        )}
      </section>
    </section>
  );
}
