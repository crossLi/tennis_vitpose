import { ArrowLeft, ActivitySquare } from "lucide-react";
import { useEffect, useMemo } from "react";
import { Link, useParams } from "react-router-dom";

import { PageBackdrop } from "@/components/PageBackdrop";
import { ResultPanel } from "@/components/ResultPanel";
import { StageTimeline } from "@/components/StageTimeline";
import { useTaskPolling } from "@/hooks/useTaskPolling";
import { useAnalysisStore } from "@/store/useAnalysisStore";

export default function TaskPage() {
  const params = useParams<{ taskId?: string }>();
  const task = useAnalysisStore((state) => state.task);
  const setTask = useAnalysisStore((state) => state.setTask);
  const error = useAnalysisStore((state) => state.error);

  useTaskPolling(params.taskId);

  useEffect(() => {
    if (!params.taskId) {
      setTask(null);
    }
  }, [params.taskId, setTask]);

  const compactId = useMemo(
    () => (params.taskId ? `${params.taskId.slice(0, 8)}...${params.taskId.slice(-4)}` : ""),
    [params.taskId],
  );

  return (
    <PageBackdrop>
      <section className="mb-5 flex flex-col gap-4 rounded-2xl border border-white/10 bg-white/[0.04] p-4 shadow-[0_24px_60px_rgba(3,9,20,0.28)] backdrop-blur-md md:mb-8 md:flex-row md:flex-wrap md:items-start md:justify-between md:gap-4 md:rounded-[32px] md:p-6">
        <div>
          <Link to="/" className="mb-3 inline-flex items-center gap-1.5 text-xs text-emerald-100 transition hover:text-white md:mb-4 md:gap-2 md:text-sm">
            <ArrowLeft className="size-3.5 md:size-4" />
            返回上传页
          </Link>
          <div className="flex items-center gap-2.5 md:gap-3">
            <div className="rounded-xl border border-cyan-300/20 bg-cyan-300/10 p-2.5 text-cyan-100 md:rounded-2xl md:p-3">
              <ActivitySquare className="size-5 md:size-6" />
            </div>
            <div>
              <p className="text-[10px] uppercase tracking-[0.24em] text-cyan-100/60 md:text-sm md:tracking-[0.32em]">任务监控</p>
              <h1 className="text-xl font-semibold text-white md:text-3xl">网球动作分析任务</h1>
            </div>
          </div>
        </div>
        <div className="rounded-2xl border border-white/10 bg-black/20 px-4 py-3 text-xs leading-6 text-slate-200 md:rounded-[28px] md:px-5 md:py-4 md:text-sm md:leading-7">
          <div>任务 ID: {compactId || "未创建"}</div>
          <div className="mt-1 text-slate-400 md:mt-0">任务页会持续轮询状态，终态时加载分析结果。</div>
        </div>
      </section>

      {error ? (
        <div className="mb-5 rounded-2xl border border-rose-400/20 bg-rose-500/10 px-4 py-3 text-xs text-rose-100 md:mb-6 md:rounded-[24px] md:px-5 md:py-4 md:text-sm">
          {error}
        </div>
      ) : null}

      {task ? (
        <>
          <StageTimeline stage={task.stage} progress={task.progress} message={task.message} />
          {["completed", "rejected"].includes(task.stage) ? <div className="mt-5 md:mt-6"><ResultPanel task={task} /></div> : null}
        </>
      ) : (
        <div className="rounded-2xl border border-white/10 bg-white/[0.04] p-6 text-center text-sm text-slate-300 shadow-[0_24px_60px_rgba(3,9,20,0.28)] backdrop-blur-md md:rounded-[32px] md:p-8 md:text-base">
          没有找到任务信息，请返回上传页重新创建任务。
        </div>
      )}
    </PageBackdrop>
  );
}
