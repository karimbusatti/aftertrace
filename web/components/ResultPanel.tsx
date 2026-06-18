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
      "ACQUIRING FRAMES",
      "ISOLATING SUBJECT",
      "MAPPING MOTION VECTORS",
      "BUILDING POINT FIELD",
      "TRACKING SUBJECT",
      "RENDERING EFFECT",
      "ENCODING OUTPUT",
    ];
    const stageIdx = Math.min(stages.length - 1, Math.floor(elapsed / 3.2));
    // Static binary columns for the surveillance backdrop.
    const cols = 7;
    const rows = 9;

    return (
      <div className="card relative overflow-hidden font-mono">
        {/* binary rain backdrop */}
        <div className="absolute inset-0 pointer-events-none select-none opacity-[0.06] leading-3 text-[10px] text-accent flex justify-between px-3">
          {Array.from({ length: cols }).map((_, c) => (
            <div key={c} className="animate-ascii-fall" style={{ animationDuration: `${3 + (c % 4) * 0.8}s`, animationDelay: `${c * 0.3}s` }}>
              {Array.from({ length: rows }).map((_, r) => (
                <div key={r}>{(c * 7 + r * 3 + stageIdx) % 2 === 0 ? "1" : "0"}</div>
              ))}
            </div>
          ))}
        </div>

        {/* CCTV corner brackets */}
        <span className="absolute top-2 left-2 w-4 h-4 border-l-2 border-t-2 border-accent/60" />
        <span className="absolute top-2 right-2 w-4 h-4 border-r-2 border-t-2 border-accent/60" />
        <span className="absolute bottom-2 left-2 w-4 h-4 border-l-2 border-b-2 border-accent/60" />
        <span className="absolute bottom-2 right-2 w-4 h-4 border-r-2 border-b-2 border-accent/60" />

        {/* sweeping scan line */}
        <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-accent to-transparent animate-scanline pointer-events-none" />

        <div className="relative p-6">
          {/* status bar */}
          <div className="flex items-center justify-between text-[10px] tracking-widest">
            <span className="flex items-center gap-1.5 text-accent">
              <span className="w-2 h-2 rounded-full bg-accent animate-pulse" /> REC
            </span>
            <span className="text-text-muted">AFTERTRACE · CAM 01</span>
          </div>

          {/* big timer */}
          <div className="text-center mt-6">
            <p className="text-accent text-4xl font-bold tabular-nums tracking-tight">
              {elapsed.toFixed(1)}<span className="text-xl">s</span>
            </p>
            <p key={stageIdx} className="text-white text-xs tracking-[0.25em] mt-2 animate-fade-in">
              {stages[stageIdx]}
              <span className="animate-pulse">_</span>
            </p>
          </div>

          {/* stepped progress */}
          <div className="flex justify-center gap-1 mt-4">
            {stages.map((_, i) => (
              <span
                key={i}
                className={`h-[3px] rounded-full transition-all duration-500 ${
                  i < stageIdx ? "w-5 bg-accent" : i === stageIdx ? "w-7 bg-accent animate-pulse" : "w-2 bg-white/15"
                }`}
              />
            ))}
          </div>

          {/* terminal fact log */}
          <div className="mt-6 pt-4 border-t border-white/10">
            <p className="text-accent/70 text-[10px] tracking-widest mb-2">{">"} INTEL</p>
            <p
              key={factIndex}
              className="text-text-secondary text-sm leading-relaxed animate-fade-in min-h-[2.5rem]"
            >
              {surveillanceFacts[factIndex]}
            </p>
          </div>
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
  return (
    <div>
      <p className="text-white text-2xl font-bold">
        {value.toLocaleString()}
      </p>
      <p className="text-text-muted text-xs mt-1">
        {label}
      </p>
    </div>
  );
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
