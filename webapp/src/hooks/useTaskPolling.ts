import { useEffect } from "react";

import { fetchAnalysisTask } from "@/lib/api";
import { useAnalysisStore } from "@/store/useAnalysisStore";

const TERMINAL_STAGES = new Set(["completed", "failed", "rejected"]);
const POLL_INTERVAL_MS = 5000;
const ERROR_RETRY_MS = 6000;

export function useTaskPolling(taskId?: string) {
  const task = useAnalysisStore((state) => state.task);
  const setTask = useAnalysisStore((state) => state.setTask);
  const setError = useAnalysisStore((state) => state.setError);

  useEffect(() => {
    if (!taskId) {
      return;
    }

    let cancelled = false;
    let timer: number | undefined;

    const poll = async () => {
      try {
        const nextTask = await fetchAnalysisTask(taskId, false);
        if (cancelled) {
          return;
        }
        if (TERMINAL_STAGES.has(nextTask.stage)) {
          const detailedTask = await fetchAnalysisTask(taskId, true);
          if (cancelled) {
            return;
          }
          setTask(detailedTask);
        } else {
          setTask(nextTask);
          timer = window.setTimeout(poll, POLL_INTERVAL_MS);
        }
      } catch (error) {
        if (!cancelled) {
          setError(error instanceof Error ? error.message : "任务状态获取失败");
          timer = window.setTimeout(poll, ERROR_RETRY_MS);
        }
      }
    };

    const shouldPoll = !task || task.taskId !== taskId || !TERMINAL_STAGES.has(task.stage);
    if (shouldPoll) {
      void poll();
    }

    return () => {
      cancelled = true;
      if (timer) {
        window.clearTimeout(timer);
      }
    };
  }, [setError, setTask, task, taskId]);
}
