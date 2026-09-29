import type { ChangeEvent } from "react";
import { Film, Sparkles, UploadCloud } from "lucide-react";

type UploadPanelProps = {
  file: File | null;
  isSubmitting: boolean;
  error: string;
  onFileChange: (file: File | null) => void;
  onSubmit: () => void;
};

export function UploadPanel({
  file,
  isSubmitting,
  error,
  onFileChange,
  onSubmit,
}: UploadPanelProps) {
  return (
    <section className="relative overflow-hidden rounded-2xl border border-white/10 bg-white/[0.04] p-4 shadow-[0_24px_60px_rgba(3,9,20,0.34)] backdrop-blur-md md:rounded-[32px] md:p-6 lg:p-8">
      <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-emerald-300/70 to-transparent" />
      <div className="mb-5 flex items-center gap-2.5 md:mb-6 md:gap-3">
        <div className="rounded-xl border border-emerald-400/20 bg-emerald-400/10 p-2.5 text-emerald-200 md:rounded-2xl md:p-3">
          <UploadCloud className="size-5 md:size-6" />
        </div>
        <div>
          <p className="text-[10px] uppercase tracking-[0.28em] text-emerald-200/70 md:text-sm md:tracking-[0.36em]">上传入口</p>
          <h2 className="text-lg font-semibold text-white md:text-2xl">提交待分析的网球视频</h2>
        </div>
      </div>

      <label className="group flex min-h-40 cursor-pointer flex-col items-center justify-center rounded-2xl border border-dashed border-white/15 bg-slate-950/40 px-4 py-6 text-center transition duration-300 hover:border-emerald-300/50 hover:bg-slate-950/55 md:min-h-56 md:rounded-[28px] md:px-6 md:py-8">
        <input
          type="file"
          accept="video/*,.avi,.mp4,.mov,.mkv,.webm"
          className="hidden"
          onChange={(event: ChangeEvent<HTMLInputElement>) => onFileChange(event.target.files?.[0] ?? null)}
        />
        <div className="mb-3 rounded-full border border-white/10 bg-white/5 p-3 text-emerald-200 transition group-hover:scale-105 group-hover:border-emerald-300/30 group-hover:bg-emerald-300/10 md:mb-4 md:p-4">
          <Film className="size-6 md:size-8" />
        </div>
        <p className="text-base font-medium text-white md:text-lg">
          {file ? file.name : "点击选择视频"}
        </p>
        <p className="mt-2 max-w-lg text-xs leading-6 text-slate-300/80 md:mt-3 md:text-sm md:leading-7">
          系统会先用多模态大模型判断它是不是&ldquo;人击打网球&rdquo;的视频，再决定是否继续做姿态、特征和专项动作分析。
        </p>
        {file ? (
          <p className="mt-4 rounded-full border border-emerald-300/20 bg-emerald-300/10 px-3 py-1.5 text-[10px] tracking-[0.2em] text-emerald-100/90 md:mt-5 md:px-4 md:py-2 md:text-xs md:tracking-[0.24em]">
            文件大小 {(file.size / 1024 / 1024).toFixed(2)} MB
          </p>
        ) : null}
      </label>

      <div className="mt-5 flex flex-col gap-3 md:flex-row md:items-center md:justify-between md:gap-4">
        <div className="rounded-xl border border-white/10 bg-black/20 px-3 py-2.5 text-xs text-slate-300/85 md:rounded-2xl md:px-4 md:py-3 md:text-sm">
          <div className="flex items-center gap-2 text-emerald-200">
            <Sparkles className="size-3.5 md:size-4" />
            <span className="leading-5">建议上传单次挥拍或单个击球片段，效果更稳定</span>
          </div>
        </div>
        <button
          type="button"
          onClick={onSubmit}
          disabled={!file || isSubmitting}
          className="inline-flex w-full items-center justify-center rounded-full border border-emerald-300/30 bg-gradient-to-r from-emerald-300 to-lime-300 px-5 py-3 text-sm font-semibold text-slate-950 transition hover:scale-[1.01] hover:shadow-[0_12px_36px_rgba(135,247,171,0.28)] disabled:cursor-not-allowed disabled:opacity-50 md:w-auto md:px-6 md:py-3"
        >
          {isSubmitting ? "正在创建分析任务..." : "开始分析"}
        </button>
      </div>

      {error ? (
        <div className="mt-4 rounded-xl border border-rose-400/20 bg-rose-500/10 px-3 py-2.5 text-xs text-rose-100 md:mt-5 md:rounded-2xl md:px-4 md:py-3 md:text-sm">
          {error}
        </div>
      ) : null}
    </section>
  );
}
