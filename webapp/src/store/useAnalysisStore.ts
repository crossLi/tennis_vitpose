import { create } from "zustand";

import type { AnalysisTask } from "@/types/analysis";

type AnalysisStore = {
  task: AnalysisTask | null;
  selectedFile: File | null;
  isSubmitting: boolean;
  error: string;
  setSelectedFile: (file: File | null) => void;
  setTask: (task: AnalysisTask | null) => void;
  setSubmitting: (isSubmitting: boolean) => void;
  setError: (error: string) => void;
  resetForNewTask: () => void;
};

export const useAnalysisStore = create<AnalysisStore>((set) => ({
  task: null,
  selectedFile: null,
  isSubmitting: false,
  error: "",
  setSelectedFile: (selectedFile) => set({ selectedFile }),
  setTask: (task) => set({ task }),
  setSubmitting: (isSubmitting) => set({ isSubmitting }),
  setError: (error) => set({ error }),
  resetForNewTask: () =>
    set({
      task: null,
      selectedFile: null,
      isSubmitting: false,
      error: "",
    }),
}));
