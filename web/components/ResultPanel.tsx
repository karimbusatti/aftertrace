"use client";

import { useEffect, useRef, useState } from "react";
import type { ProcessResponse } from "@/lib/api";
import { getDownloadUrl } from "@/lib/api";
import { surveillanceFacts, getTrackingLevel } from "@/lib/copy";

interface ResultPanelProps {
  result: ProcessResponse | null;
  error: string | null;
  isLoading: boolean;
  onOpenTips: () => void;
}

export function ResultPanel({ result, error, isLoading, onOpenTips }: ResultPanelProps) {
  // Live elapsed timer while the video is processing.
  const [elapsed, setElapsed] = useState(0);
  const startRef = useRef<number | null>(null);

  useEffect(() => {
    if (!isLoading) return;
    startRef.current = Date.now();
    setElapsed(0);
    const id = setInterval(() => {
      if (startRef.current !== null) {
        setElapsed((Date.now() - startRef.current) / 1000);
      }
    }, 100);
    return () => clearInterval(id);
  }, [isLoading]);

  if (isLoading) {
    const factIndex = Math.floor(elapsed / 4.5) % surveillanceFacts.length;
    const stages = [
      "reading frames",
      "isolating subject",
      "mapping motion",
      "building points",
      "tracking subject",
      "rendering",
      "encoding",
    ];
    const stageIdx = Math.min(stages.length - 1, Math.floor(elapsed / 3.2));
    const pct = ((stageIdx + 1) / stages.length) * 100;

    return (
      <div className="card relative overflow-hidden px-8 py-12 text-center">
        {/* one hairline sweeping the panel - the only motion besides the timer */}
        <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-white/30 to-transparent animate-scanline pointer-events-none" />

        {/* thin minimal ring */}
        <div className="w-12 h-12 mx-auto rounded-full border border-white/10 border-t-white/70 animate-spin [animation-duration:1.4s]" />

        {/* big light timer */}
        <p className="text-white text-5xl font-light tabular-nums mt-8 tracking-tight">
          {elapsed.toFixed(1)}
          <span className="text-lg text-text-muted ml-1 font-normal">s</span>
        </p>

        <p key={stageIdx} className="text-text-muted text-[11px] tracking-[0.3em] uppercase mt-3 animate-fade-in">
          {stages[stageIdx]}
        </p>

        {/* hairline progress */}
        <div className="h-px bg-white/10 rounded-full mt-7 overflow-hidden max-w-[200px] mx-auto">
          <div className="h-full bg-accent transition-all duration-700 ease-out" style={{ width: `${pct}%` }} />
        </div>

        {/* one quiet fact */}
        <div className="mt-9">
          <p className="text-text-muted/60 text-[10px] uppercase tracking-[0.3em] mb-2.5">did you know</p>
          <p
            key={factIndex}
            className="text-text-secondary text-sm leading-relaxed animate-fade-in min-h-[2.5rem] max-w-sm mx-auto"
          >
            {surveillanceFacts[factIndex]}
          </p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="card p-6 border-danger/20">
        <div className="flex items-start gap-3">
          <span className="text-danger text-xl">✕</span>
          <div>
            <p className="text-white font-medium">{error}</p>
            <p className="text-text-muted text-sm mt-1">
              try a different video or format
            </p>
          </div>
        </div>
      </div>
    );
  }

  if (!result) return null;

  const metadata = result.metadata;
  const trackability = metadata?.trackability_score ?? 0;
  const videoUrl = getDownloadUrl(result.job_id);

  return (
    <div className="space-y-4">
      {/* Video */}
      <div className="card overflow-hidden">
        <video
          src={videoUrl}
          controls
          autoPlay
          loop
          muted
          playsInline
          className="w-full aspect-video bg-black"
        />
      </div>

      {/* Stats */}
      <div className="card p-5">
        <div className="flex items-center justify-between mb-3">
          <span className="text-text-muted text-sm">analysis results</span>
          <TrackabilityBadge score={trackability} />
        </div>

        <p className="text-text-secondary text-sm leading-relaxed mb-4">
          {getTrackingLevel(trackability).message}
        </p>

        {metadata?.segments_applied && metadata.segments_applied.length > 0 && (
          <div className="mb-6 p-3 bg-white/5 rounded-lg">
             <p className="text-text-muted text-[10px] uppercase tracking-wider mb-2">Sequence</p>
             <div className="flex h-1.5 rounded-full overflow-hidden gap-0.5">
               {metadata.segments_applied.map((seg, i) => (
                 <div 
                   key={i} 
                   className="flex-1 bg-accent/40 first:bg-accent/60 last:bg-accent/80"
                   title={seg.effect}
                 />
               ))}
             </div>
             <div className="flex justify-between mt-1.5">
               {metadata.segments_applied.map((seg, i) => (
                 <span key={i} className="text-[9px] text-text-muted truncate max-w-[50px] block">
                   {seg.effect.replace('_', ' ')}
                 </span>
               ))}
             </div>
          </div>
        )}
        
        <div className="grid grid-cols-3 gap-4 text-center">
          <StatBlock 
            value={metadata?.total_points_spawned ?? 0} 
            label="points tracked" 
          />
          <StatBlock 
            value={metadata?.frames_processed ?? 0} 
            label="frames" 
          />
          <StatBlock
            value={metadata?.people_detected ?? 0}
            label="people detected"
          />
        </div>

        {metadata?.processing_time_seconds != null && (
          <p className="text-text-muted text-xs text-center mt-4 font-mono">
            rendered in {metadata.processing_time_seconds.toFixed(1)}s
          </p>
        )}
      </div>

      {/* Actions */}
      <div className="flex gap-3">
        <a
          href={videoUrl}
          download
          className="btn-primary flex-1 text-center"
        >
          download
        </a>
        <button
          onClick={onOpenTips}
          className="btn-secondary"
        >
          how to hide
        </button>
      </div>
    </div>
  );
}

function StatBlock({ value, label }: { value: number; label: string }) {
  const display = useCountUp(value);
  return (
    <div>
      <p className="text-white text-2xl font-bold tabular-nums">
        {display.toLocaleString()}
      </p>
      <p className="text-text-muted text-xs mt-1">
        {label}
      </p>
    </div>
  );
}

/** Count up to `target` over ~0.9s with ease-out; skipped for reduced motion. */
function useCountUp(target: number): number {
  const [display, setDisplay] = useState(target);
  useEffect(() => {
    if (typeof window === "undefined" || target <= 0) {
      setDisplay(target);
      return;
    }
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      setDisplay(target);
      return;
    }
    const duration = 900;
    const start = performance.now();
    let raf: number;
    const tick = (now: number) => {
      const t = Math.min(1, (now - start) / duration);
      const eased = 1 - Math.pow(1 - t, 3);
      setDisplay(Math.round(target * eased));
      if (t < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [target]);
  return display;
}

function TrackabilityBadge({ score }: { score: number }) {
  const getLevel = (s: number) => {
    if (s >= 70) return { label: "high", color: "text-danger", bg: "bg-danger/10" };
    if (s >= 40) return { label: "medium", color: "text-warning", bg: "bg-warning/10" };
    return { label: "low", color: "text-success", bg: "bg-success/10" };
  };

  const level = getLevel(score);

  return (
    <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full ${level.bg}`}>
      <span className="text-white font-bold">{score}%</span>
      <span className={`text-xs font-medium ${level.color}`}>
        {level.label} trackability
      </span>
    </div>
  );
}
