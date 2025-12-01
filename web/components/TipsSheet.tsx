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
        className="absolute inset-0 bg-black/80 backdrop-blur-sm"
        onClick={onClose}
      />
      
      <div
        ref={sheetRef}
        className="absolute inset-x-0 bottom-0 max-h-[85vh] bg-black border-t border-white/10 overflow-y-auto"
      >
        <div className="sticky top-0 bg-black/90 backdrop-blur-sm border-b border-white/5 px-6 py-4 flex items-center justify-between z-10">
          <div>
            <h2 className="text-white text-lg font-light">
              Reduce your footprint
            </h2>
            <p className="text-text-muted text-[11px] font-mono uppercase tracking-widest mt-1">
              Privacy techniques
            </p>
          </div>
          <button
            onClick={onClose}
            className="w-8 h-8 flex items-center justify-center text-text-secondary hover:text-white transition-colors"
          >
            ×
          </button>
        </div>

        <div className="p-6 space-y-8">
          {/* Personalized tips based on score */}
          {personalizedTips && trackingLevel && (
            <div className="border border-white/10 p-5">
              <div className="flex items-center gap-3 mb-4">
                <div className={`w-2 h-2 rounded-full ${
                  trackingLevel.level === "high" ? "bg-danger" :
                  trackingLevel.level === "medium" ? "bg-warning" : "bg-success"
                }`} />
                <span className="text-[11px] font-mono uppercase tracking-widest text-text-muted">
                  Your score: {trackabilityScore}%
                </span>
              </div>
              <p className="text-white text-sm mb-4">
                {trackingLevel.message}
              </p>
              <ul className="space-y-2">
                {personalizedTips.map((tip, idx) => (
                  <li key={idx} className="text-text-secondary text-sm flex items-start gap-2">
                    <span className="text-text-muted font-mono text-[10px] mt-0.5">
                      {String(idx + 1).padStart(2, "0")}
                    </span>
                    <span>{tip}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* General sections */}
          {tipsSections.map((section) => (
            <div key={section.id}>
              <h3 className="text-text-muted text-[11px] font-mono uppercase tracking-widest mb-4">
                {section.title}
              </h3>
              <ul className="space-y-3">
                {section.tips.map((tip, idx) => (
                  <li key={idx} className="text-text-secondary text-sm flex items-start gap-3">
                    <span className="text-text-muted font-mono text-[10px] mt-0.5 shrink-0">
                      {String(idx + 1).padStart(2, "0")}
                    </span>
                    <span className="leading-relaxed">{tip}</span>
                  </li>
                ))}
              </ul>
            </div>
          ))}

          <div className="border-t border-white/5 pt-6">
            <p className="text-text-muted text-[11px] font-mono text-center uppercase tracking-widest">
              Your data was deleted after processing
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
