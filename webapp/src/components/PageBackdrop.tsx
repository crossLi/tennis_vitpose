import type { ReactNode } from "react";

type PageBackdropProps = {
  children: ReactNode;
};

export function PageBackdrop({ children }: PageBackdropProps) {
  return (
    <div className="min-h-screen min-h-[100dvh] bg-white text-[#333333]">
      <main className="relative z-10 mx-auto w-full max-w-3xl px-4 py-5 md:max-w-4xl md:px-6 md:py-8 lg:max-w-5xl lg:py-10 safe-bottom">
        {children}
      </main>
    </div>
  );
}
