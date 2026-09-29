import { useEffect } from "react";
import { useParams } from "react-router-dom";

import { PageBackdrop } from "@/components/PageBackdrop";
import { ResultPanel } from "@/components/ResultPanel";
import { useTaskPolling } from "@/hooks/useTaskPolling";
import { useAnalysisStore } from "@/store/useAnalysisStore";

export default function ReportPage() {
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

  return (
    <PageBackdrop>
      {error ? (
        <div className="mb-5 rounded-2xl border border-[#FEE2E2] bg-[#FFF1F2] px-4 py-3 text-xs text-[#B91C1C] md:mb-6 md:rounded-[24px] md:px-5 md:py-4 md:text-sm">
          {error}
        </div>
      ) : null}

      {task ? (
        <>
          {["completed", "rejected"].includes(task.stage) ? (
            <ResultPanel task={task} />
          ) : (
            <div className="report-card text-center text-sm text-[#666666] md:text-base">
              报告正在生成中，请稍后刷新页面。
            </div>
          )}
        </>
      ) : (
        <div className="report-card text-center text-sm text-[#666666] md:text-base">
          没有找到对应的分析报告。
        </div>
      )}
    </PageBackdrop>
  );
}
