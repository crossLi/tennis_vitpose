import { ArrowRight, BarChart3, Cpu, ShieldCheck } from "lucide-react";
import { useNavigate } from "react-router-dom";

import { PageBackdrop } from "@/components/PageBackdrop";
import { UploadPanel } from "@/components/UploadPanel";
import { createAnalysisTask } from "@/lib/api";
import { useAnalysisStore } from "@/store/useAnalysisStore";

export default function LandingPage() {
  const navigate = useNavigate();
  const task = useAnalysisStore((state) => state.task);
  const selectedFile = useAnalysisStore((state) => state.selectedFile);
  const isSubmitting = useAnalysisStore((state) => state.isSubmitting);
  const error = useAnalysisStore((state) => state.error);
  const setSelectedFile = useAnalysisStore((state) => state.setSelectedFile);
  const setTask = useAnalysisStore((state) => state.setTask);
  const setSubmitting = useAnalysisStore((state) => state.setSubmitting);
  const setError = useAnalysisStore((state) => state.setError);

  const heroMetrics = [
    { label: "预判阶段", value: "大模型先判别是否是人打网球" },
    { label: "姿态链路", value: "RTMPose + 特征提取 + 摘要生成" },
    { label: "结果形式", value: "动态状态 + 视频对照 + 动作分析结论" },
  ];

  const handleSubmit = async () => {
    if (!selectedFile) {
      setError("请先选择一个视频文件");
      return;
    }

    setSubmitting(true);
    setError("");
    try {
      const createdTask = await createAnalysisTask(selectedFile);
      setTask(createdTask);
      navigate(`/task/${createdTask.taskId}`);
    } catch (submitError) {
      setError(submitError instanceof Error ? submitError.message : "创建任务失败");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <PageBackdrop>
      <section className="mb-6 space-y-5 xl:mb-8 xl:grid xl:gap-8 xl:grid-cols-[1.1fr_0.9fr]">
        <div className="space-y-5 xl:space-y-6">
          <div className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/[0.04] px-3 py-1.5 text-[10px] uppercase tracking-[0.24em] text-emerald-200/80 md:px-4 md:py-2 md:text-xs md:tracking-[0.32em]">
            <ShieldCheck className="size-3.5 md:size-4" />
            Tennis Motion Intelligence
          </div>
          <div>
            <h1 className="font-semibold tracking-tight text-white leading-tight">
              上传网球视频，AI 教练自动分析你的动作
            </h1>
            <p className="mt-4 text-sm leading-7 text-slate-300/82 md:text-base md:leading-8">
              系统会先判断这是不是&ldquo;真人击打网球&rdquo;的视频，如果不是，直接返回无法分析；如果是，就继续做姿态估计、专项特征提取、摘要整理，再由多模态大模型输出更像教练视角的动作结论。
            </p>
          </div>
          <div className="grid grid-cols-1 gap-3 md:gap-4 md:grid-cols-3">
            {heroMetrics.map((item) => (
              <div key={item.label} className="h5-card md:rounded-[28px] md:p-5">
                <p className="text-[10px] uppercase tracking-[0.2em] text-slate-400 md:text-xs md:tracking-[0.24em]">{item.label}</p>
                <p className="mt-2 text-xs leading-6 text-slate-200 md:mt-3 md:text-sm md:leading-7">{item.value}</p>
              </div>
            ))}
          </div>
        </div>

        <div className="mt-5 grid gap-3 self-start xl:mt-0 xl:gap-4">
          <div className="h5-card bg-gradient-to-br from-white/[0.08] to-white/[0.03] md:rounded-[32px] md:p-6">
            <div className="mb-4 flex items-center gap-2.5 md:mb-5 md:gap-3">
              <Cpu className="size-4.5 text-cyan-200 md:size-5" />
              <h2 className="font-semibold text-white">完整处理链路</h2>
            </div>
            <div className="space-y-3 md:space-y-4">
              {[
                "多模态大模型先判断视频里是不是人在进行网球击打",
                "RTMPose 生成姿态可视化视频和逐帧关键点",
                "提取肩髋分离、旋转速度、站位等网球特征",
                "再由多模态大模型把视频和结构化结果组合成动作分析",
              ].map((text, index) => (
                <div key={text} className="flex items-start gap-2.5 h5-card-sm text-xs text-slate-200 md:gap-3 md:rounded-[24px] md:px-4 md:py-3 md:text-sm">
                  <div className="mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded-full border border-emerald-300/20 bg-emerald-300/10 text-[10px] text-emerald-100 md:h-7 md:w-7 md:text-xs">
                    {index + 1}
                  </div>
                  <p className="leading-6 md:leading-7">{text}</p>
                </div>
              ))}
            </div>
          </div>

          <div className="h5-card md:rounded-[32px] md:p-6">
            <div className="mb-2.5 flex items-center gap-2.5 md:mb-3 md:gap-3">
              <BarChart3 className="size-4.5 text-amber-200 md:size-5" />
              <h2 className="font-semibold text-white">当前任务入口</h2>
            </div>
            {task ? (
              <button
                type="button"
                onClick={() => navigate(`/task/${task.taskId}`)}
                className="inline-flex items-center gap-1.5 text-xs text-emerald-100 transition hover:text-white md:gap-2 md:text-sm"
              >
                查看任务 {task.taskId.slice(0, 8)}
                <ArrowRight className="size-3.5 md:size-4" />
              </button>
            ) : (
              <p className="text-xs leading-6 text-slate-300/80 md:text-sm md:leading-7">上传后会自动跳转到任务详情页，并开始轮询显示分析状态。</p>
            )}
          </div>
        </div>
      </section>

      <UploadPanel
        file={selectedFile}
        isSubmitting={isSubmitting}
        error={error}
        onFileChange={setSelectedFile}
        onSubmit={handleSubmit}
      />
    </PageBackdrop>
  );
}
