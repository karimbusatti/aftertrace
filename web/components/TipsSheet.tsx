"use client";

import { useEffect, useCallback } from "react";
import { tipsSections, miscCopy, getPersonalizedTips, getTrackingLevel } from "@/lib/copy";

interface TipsSheetProps {
  isOpen: boolean;
  onClose: () => void;
  trackabilityScore?: number;
}

export function TipsSheet({ isOpen, onClose, trackabilityScore }: TipsSheetProps) {
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

  const personalizedTips = trackabilityScore !== undefined 
    ? getPersonalizedTips(trackabilityScore) 
    : null;
  const trackingLevel = trackabilityScore !== undefined 
    ? getTrackingLevel(trackabilityScore) 
    : null;

  return (
    <div className="fixed inset-0 z-50 flex items-end justify-center">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/70 backdrop-blur-sm animate-fade-in"
        onClick={onClose}
        aria-hidden="true"
      />

      {/* Sheet */}
      <div
        className="relative w-full max-w-lg max-h-[85dvh] bg-surface-raised rounded-t-3xl 
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
          <p className="text-text-secondary text-sm mt-1 font-mono">
            {miscCopy.tipsSheetSubtitle}
          </p>
        </div>

        {/* Content */}
        <div className="overflow-y-auto max-h-[calc(85dvh-140px)] px-6 py-5 space-y-6">
          {/* Personalized tips based on score */}
          {personalizedTips && trackingLevel && (
            <div className="space-y-3">
              <div className="flex items-center gap-2">
                <div className={`w-2 h-2 rounded-full ${
                  trackingLevel === 'high' ? 'bg-danger animate-pulse' :
                  trackingLevel === 'medium' ? 'bg-warning' : 'bg-success'
                }`} />
                <h3 className="font-display font-semibold text-text-primary uppercase tracking-wider text-xs">
                  based on your clip
                </h3>
                {trackabilityScore !== undefined && (
                  <span className="ml-auto text-text-muted text-xs font-mono">
                    score: {trackabilityScore}/100
                  </span>
                )}
              </div>
              
              <div className={`p-4 rounded-xl border ${
                trackingLevel === 'high' ? 'bg-danger/5 border-danger/20' :
                trackingLevel === 'medium' ? 'bg-warning/5 border-warning/20' : 
                'bg-success/5 border-success/20'
              }`}>
                <ul className="space-y-2">
                  {personalizedTips.map((tip, index) => (
                    <li key={index} className="text-sm text-text-secondary leading-relaxed flex gap-2">
                      <span className="text-text-muted">›</span>
                      <span>{tip}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          )}

          {/* General tips sections */}
          {tipsSections.map((section) => (
            <TipsSection key={section.id} section={section} />
          ))}

          {/* Closing note */}
          <div className="pt-4 border-t border-white/5">
            <p className="text-text-muted text-sm leading-relaxed font-mono">
              this isn&apos;t paranoia. it&apos;s literacy. 
              <br />
              do what feels right for you.
            </p>
          </div>
        </div>

        {/* Close button */}
        <div className="px-6 py-4 border-t border-white/5 bg-surface-raised">
          <button
            onClick={onClose}
            className="w-full py-3 px-4 rounded-xl bg-white/5 hover:bg-white/10 
                       text-text-primary text-sm font-medium transition-colors"
          >
            close
          </button>
        </div>
      </div>
    </div>
  );
}

function TipsSection({ section }: { section: (typeof tipsSections)[number] }) {
  return (
    <div className="space-y-3">
      {/* Section header */}
      <div className="flex items-center gap-2">
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
