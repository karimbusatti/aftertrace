"use client";

import { useEffect, useCallback } from "react";
import { tipsSections, miscCopy } from "@/lib/copy";

interface TipsSheetProps {
  isOpen: boolean;
  onClose: () => void;
}

export function TipsSheet({ isOpen, onClose }: TipsSheetProps) {
  // Close on escape key
  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    },
    [onClose]
  );

  useEffect(() => {
    if (isOpen) {
      document.addEventListener("keydown", handleKeyDown);
      document.body.style.overflow = "hidden";
    }
    return () => {
      document.removeEventListener("keydown", handleKeyDown);
      document.body.style.overflow = "";
    };
  }, [isOpen, handleKeyDown]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-end justify-center">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/60 backdrop-blur-sm animate-fade-in"
        onClick={onClose}
        aria-hidden="true"
      />

      {/* Sheet */}
      <div
        className="relative w-full max-w-lg max-h-[85dvh] bg-surface-base rounded-t-3xl 
                   border-t border-x border-white/10 overflow-hidden
                   animate-slide-up"
        role="dialog"
        aria-modal="true"
        aria-labelledby="tips-sheet-title"
      >
        {/* Handle */}
        <div className="flex justify-center pt-3 pb-2">
          <div className="w-10 h-1 rounded-full bg-white/20" />
        </div>

        {/* Header */}
        <div className="px-6 pb-4 border-b border-white/5">
          <h2
            id="tips-sheet-title"
            className="font-display font-bold text-xl text-text-primary"
          >
            {miscCopy.tipsSheetTitle}
          </h2>
          <p className="text-text-secondary text-sm mt-1">
            {miscCopy.tipsSheetSubtitle}
          </p>
        </div>

        {/* Content */}
        <div className="overflow-y-auto max-h-[calc(85dvh-120px)] px-6 py-5 space-y-6">
          {tipsSections.map((section) => (
            <TipsSection key={section.id} section={section} />
          ))}

          {/* Closing note */}
          <div className="pt-4 border-t border-white/5">
            <p className="text-text-muted text-sm leading-relaxed">
              none of this is paranoia. it&apos;s just knowing how the game works. 
              do what feels right for you.
            </p>
          </div>
        </div>

        {/* Close button */}
        <div className="px-6 py-4 border-t border-white/5 bg-surface-base">
          <button
            onClick={onClose}
            className="w-full py-3 px-4 rounded-xl bg-white/5 hover:bg-white/10 
                       text-text-primary text-sm font-medium transition-colors"
          >
            got it
          </button>
        </div>
      </div>
    </div>
  );
}

function TipsSection({ section }: { section: typeof tipsSections[number] }) {
  return (
    <div>
      {/* Section header */}
      <div className="flex items-center gap-2 mb-3">
        <div className="w-1.5 h-1.5 rounded-full bg-accent" />
        <h3 className="font-display font-semibold text-text-primary uppercase tracking-wider text-xs">
          {section.title}
        </h3>
      </div>

      {/* Tips list */}
      <ul className="space-y-2 pl-3 border-l border-white/5">
        {section.tips.map((tip, index) => (
          <li key={index} className="text-sm text-text-secondary leading-relaxed">
            {tip}
          </li>
        ))}
      </ul>
    </div>
  );
}

