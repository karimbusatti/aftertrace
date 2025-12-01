"use client";

import { useEffect, useRef } from "react";
import { tipsSections, getPersonalizedTips, getTrackingLevel } from "@/lib/copy";

interface TipsSheetProps {
  isOpen: boolean;
  onClose: () => void;
  trackabilityScore?: number;
}

export function TipsSheet({ isOpen, onClose, trackabilityScore }: TipsSheetProps) {
  const sheetRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };

    if (isOpen) {
      document.addEventListener("keydown", handleKeyDown);
      document.body.style.overflow = "hidden";
    }

    return () => {
      document.removeEventListener("keydown", handleKeyDown);
      document.body.style.overflow = "";
    };
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  const personalizedTips = trackabilityScore !== undefined 
    ? getPersonalizedTips(trackabilityScore) 
    : null;
  const trackingLevel = trackabilityScore !== undefined 
    ? getTrackingLevel(trackabilityScore) 
    : null;

  return (
    <div className="fixed inset-0 z-[9999]">
      <div 
        className="absolute inset-0 bg-black/90"
        onClick={onClose}
      />
      
      <div
        ref={sheetRef}
        className="absolute inset-x-0 bottom-0 max-h-[80vh] bg-black border-t border-white/5 overflow-y-auto"
      >
        {/* Header */}
        <div className="sticky top-0 bg-black/95 backdrop-blur-sm border-b border-white/[0.03] px-5 py-3.5 flex items-center justify-between z-10">
          <span className="text-white text-sm font-light">
            Reduce your footprint
          </span>
          <button
            onClick={onClose}
            className="w-6 h-6 flex items-center justify-center text-text-secondary hover:text-white transition-colors"
          >
            ×
          </button>
        </div>

        <div className="p-5 space-y-6">
          {/* Personalized section */}
          {personalizedTips && trackingLevel && (
            <div className="border border-white/[0.06] p-4">
              <div className="flex items-center gap-2 mb-3">
                <div className={`w-1.5 h-1.5 rounded-full ${
                  trackingLevel.level === "high" ? "bg-danger" :
                  trackingLevel.level === "medium" ? "bg-warning" : "bg-success"
                }`} />
                <span className="text-[9px] font-mono uppercase tracking-[0.15em] text-text-muted">
                  Your score: {trackabilityScore}%
                </span>
              </div>
              <p className="text-white text-xs mb-3">
                {trackingLevel.message}
              </p>
              <ul className="space-y-1.5">
                {personalizedTips.map((tip, idx) => (
                  <li key={idx} className="text-text-secondary text-[11px] flex items-start gap-2">
                    <span className="text-text-muted font-mono text-[9px] mt-0.5 w-4">
                      {idx + 1}.
                    </span>
                    <span className="leading-relaxed">{tip}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* General tips */}
          {tipsSections.map((section) => (
            <div key={section.id}>
              <h3 className="text-text-muted text-[9px] font-mono uppercase tracking-[0.2em] mb-3">
                {section.title}
              </h3>
              <ul className="space-y-2">
                {section.tips.map((tip, idx) => (
                  <li key={idx} className="text-text-secondary text-[11px] leading-relaxed">
                    {tip}
                  </li>
                ))}
              </ul>
            </div>
          ))}

          <div className="border-t border-white/[0.03] pt-4">
            <p className="text-text-muted text-[9px] font-mono text-center uppercase tracking-[0.15em]">
              Your data was deleted
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
