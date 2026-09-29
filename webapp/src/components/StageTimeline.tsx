import { CheckCircle2, CircleDashed, LoaderCircle, Video, Waves, XCircle } from "lucide-react";

import type { AnalysisStage } from "@/types/analysis";

const STAGES: Array<{ key: AnalysisStage; label: string; hint: string }> = [
  { key: "queued", label: "任务创建", hint: "视频已经上传到服务端" },
  { key: "classifying", label: "网球视频判别", hint: "多模态大模型正在判断是否为真人击打网球场景" },
  { key: "generating_pose_video", label: "姿态可视化", hint: "正在生成姿态视频与关键点序列" },
  { key: "extracting_features", label: "特征提取", hint: "正在提取肩髋分离、旋转速度、站位等特征" },
  { key: "summarizing_features", label: "摘要整理", hint: "正在生成适合大模型进一步分析的摘要" },
  { key: "analyzing_with_kimi", label: "专项分析", hint: "多模态大模型正在结合视频与结构化特征给出动作结论" },
  { key: "completed", label: "完成", hint: "分析结果已准备好" },
];

const STAGE_ORDER = STAGES.map((item) => item.key);

type StageTimelineProps = {
  stage: AnalysisStage;
  progress: number;
  message: string;
};

function stageStatus(currentStage: AnalysisStage, targetStage: AnalysisStage) {
  if (currentStage === "failed") {
    return "failed";
  }
  if (currentStage === "rejected") {
    return targetStage === "classifying" ? "failed" : "idle";
  }
  const currentIndex = STAGE_ORDER.indexOf(currentStage);
  const targetIndex = STAGE_ORDER.indexOf(targetStage);
  if (currentIndex === -1 || targetIndex === -1) {
    return "idle";
  }
  if (currentIndex > targetIndex) {
    return "done";
  }
  if (currentIndex === targetIndex) {
    return "active";
  }
  return "idle";
}

function statusIcon(status: string) {
  if (status === "done") {
    return <CheckCircle2 className="size-4 md:size-5" />;
  }
  if (status === "active") {
    return <LoaderCircle className="size-4 md:size-5 animate-spin" />;
  }
  if (status === "failed") {
    return <XCircle className="size-4 md:size-5" />;
  }
  return <CircleDashed className="size-4 md:size-5" />;
}

export function StageTimeline({ stage, progress, message }: StageTimelineProps) {
  return (
    <section className="rounded-2xl border border-white/10 bg-white/[0.04] p-4 shadow-[0_24px_60px_rgba(3,9,20,0.28)] backdrop-blur-md md:rounded-[32px] md:p-6 lg:p-8">
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3 md:mb-5 md:gap-4">
        <div>
          <p className="text-[10px] uppercase tracking-[0.24em] text-cyan-100/60 md:text-sm md:tracking-[0.32em]">任务状态</p>
          <h2 className="mt-1.5 text-lg font-semibold text-white md:mt-2 md:text-2xl">完整分析链路正在推进</h2>
        </div>
        <div className="inline-flex items-center gap-1.5 rounded-full border border-cyan-300/20 bg-cyan-300/10 px-3 py-1.5 text-xs text-cyan-100 md:gap-2 md:px-4 md:py-2 md:text-sm">
          <Waves className="size-3.5 md:size-4" />
          <span>{Math.round(progress * 100)}%</span>
        </div>
      </div>

      <div className="mb-4 overflow-hidden rounded-full bg-white/5 md:mb-6">
        <div
          className="h-1.5 rounded-full bg-gradient-to-r from-cyan-300 via-emerald-300 to-lime-300 transition-all duration-500 md:h-2"
          style={{ width: `${Math.max(progress * 100, 6)}%` }}
        />
      </div>

      <div className="mb-4 rounded-2xl border border-white/10 bg-slate-950/45 p-4 text-xs leading-6 text-slate-200 md:mb-6 md:rounded-[28px] md:p-5 md:text-sm md:leading-7">
        <div className="mb-1.5 flex items-center gap-1.5 text-cyan-200 md:mb-2 md:gap-2">
          <Video className="size-3.5 md:size-4" />
          <span>当前提示</span>
        </div>
        <p>{message}</p>
      </div>

      <div className="grid gap-2.5 md:gap-4">
        {STAGES.map((item) => {
          const status = stageStatus(stage, item.key);
          return (
            <div
              key={item.key}
              className={[
                "flex items-start gap-2.5 rounded-xl border px-3 py-3 transition md:gap-4 md:rounded-[24px] md:px-4 md:py-4",
                status === "done" ? "border-emerald-300/20 bg-emerald-300/10 text-emerald-50" : "",
                status === "active" ? "border-cyan-300/30 bg-cyan-300/10 text-cyan-50" : "",
                status === "idle" ? "border-white/8 bg-white/[0.02] text-slate-300/70" : "",
                status === "failed" ? "border-rose-400/25 bg-rose-500/10 text-rose-100" : "",
              ].join(" ")}
            >
              <div className="mt-0.5 shrink-0">{statusIcon(status)}</div>
              <div className="min-w-0 flex-1">
                <p className="text-sm font-medium md:text-base">{item.label}</p>
                <p className="mt-0.5 text-[11px] opacity-80 leading-5 md:mt-1 md:text-sm md:leading-6">{item.hint}</p>
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}
