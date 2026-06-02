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
        className="absolute inset-x-0 bottom-0 max-h-[85vh] bg-surface-raised border-t border-white/10 rounded-t-3xl overflow-y-auto"
      >
        {/* Header */}
        <div className="sticky top-0 bg-surface-raised/95 backdrop-blur-sm border-b border-white/5 px-6 py-5 flex items-center justify-between z-10">
          <div>
            <h2 className="text-white text-xl font-semibold">
              how to be less trackable
            </h2>
            <p className="text-text-muted text-sm mt-1">
              practical tips for digital privacy
            </p>
          </div>
          <button
            onClick={onClose}
            className="w-10 h-10 rounded-full bg-white/5 flex items-center justify-center text-text-secondary hover:text-white hover:bg-white/10 transition-colors"
          >
            ✕
          </button>
        </div>

        <div className="p-6 space-y-8">
          {/* Personalized tips */}
          {personalizedTips && trackingLevel && (
            <div className="bg-accent/5 border border-accent/20 rounded-2xl p-5">
              <div className="flex items-center gap-3 mb-4">
                <div className={`w-3 h-3 rounded-full ${
                  trackingLevel.level === "high" ? "bg-danger" :
                  trackingLevel.level === "medium" ? "bg-warning" : "bg-success"
                }`} />
                <span className="text-white font-medium">
                  your score: {trackabilityScore}%
                </span>
              </div>
              <p className="text-text-secondary mb-4">
                {trackingLevel.message}
              </p>
              <ul className="space-y-3">
                {personalizedTips.map((tip, idx) => (
                  <li key={idx} className="flex items-start gap-3 text-text-secondary">
                    <span className="text-accent font-mono text-sm mt-0.5">
                      {String(idx + 1).padStart(2, "0")}
                    </span>
                    <span>{tip}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* General tips sections */}
          {tipsSections.map((section) => (
            <div key={section.id}>
              <div className="flex items-center gap-3 mb-1">
                <span className="text-xl" aria-hidden>{section.icon}</span>
                <h3 className="text-white font-semibold text-lg">
                  {section.title}
                </h3>
              </div>
              <p className="text-text-muted text-sm mb-4 leading-relaxed">
                {section.intro}
              </p>
              <ul className="space-y-2.5">
                {section.tips.map((tip, idx) => (
                  <li
                    key={idx}
                    className="flex items-start gap-3 text-text-secondary bg-white/[0.02] rounded-xl px-3 py-2.5 border border-white/5"
                  >
                    <ImpactDot impact={tip.impact} />
                    <span className="leading-relaxed text-sm">{tip.text}</span>
                  </li>
                ))}
              </ul>
            </div>
          ))}

          {/* Footer note */}
          <div className="border-t border-white/5 pt-6 text-center space-y-2">
            <div className="flex items-center justify-center gap-4 text-[11px] text-text-muted">
              <span className="flex items-center gap-1.5"><Dot color="bg-danger" /> high impact</span>
              <span className="flex items-center gap-1.5"><Dot color="bg-warning" /> medium</span>
              <span className="flex items-center gap-1.5"><Dot color="bg-text-muted" /> nice to have</span>
            </div>
            <p className="text-text-muted text-sm">
              your video was deleted from our servers after processing
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

function Dot({ color }: { color: string }) {
  return <span className={`inline-block w-2 h-2 rounded-full ${color}`} />;
}

function ImpactDot({ impact }: { impact: "high" | "medium" | "low" }) {
  const color =
    impact === "high" ? "bg-danger" : impact === "medium" ? "bg-warning" : "bg-text-muted";
  return (
    <span
      className={`mt-1.5 shrink-0 w-2 h-2 rounded-full ${color}`}
      title={`${impact} impact`}
      aria-label={`${impact} impact`}
    />
  );
}
