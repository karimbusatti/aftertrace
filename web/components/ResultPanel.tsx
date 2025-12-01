"use client";

import type { ProcessResponse } from "@/lib/api";

interface ResultPanelProps {
  result: ProcessResponse | null;
  error: string | null;
  isLoading: boolean;
  onOpenTips: () => void;
}

export function ResultPanel({ result, error, isLoading, onOpenTips }: ResultPanelProps) {
  if (isLoading) {
    return (
      <div className="card p-8 text-center">
        <div className="inline-flex items-center gap-3">
          <div className="w-4 h-4 border border-white/30 border-t-white rounded-full animate-spin" />
          <span className="text-text-secondary text-sm">
            Analyzing frame by frame...
          </span>
        </div>
        <p className="text-text-muted text-[11px] mt-3 font-mono">
          Processing takes about 30 seconds
        </p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="card p-6 border-danger/30">
        <div className="flex items-start gap-3">
          <span className="text-danger text-sm">Error</span>
          <div>
            <p className="text-white text-sm">{error}</p>
            <p className="text-text-muted text-[11px] mt-1">
              Try a different video or format
            </p>
          </div>
        </div>
      </div>
    );
  }

  if (!result) return null;

  const metadata = result.metadata;
  const trackability = metadata?.trackability_score ?? 0;

  return (
    <div className="space-y-4">
      <div className="card overflow-hidden">
        <video
          src={result.video_url}
          controls
          autoPlay
          loop
          muted
          playsInline
          className="w-full aspect-video bg-black"
        />
      </div>

      <div className="card p-5">
        <div className="flex items-center justify-between mb-4">
          <span className="text-text-muted text-[11px] font-mono uppercase tracking-widest">
            Analysis
          </span>
          <TrackabilityBadge score={trackability} />
        </div>
        
        <div className="grid grid-cols-3 gap-4 text-center">
          <div>
            <p className="text-white text-2xl font-light font-mono">
              {metadata?.points_tracked ?? 0}
            </p>
            <p className="text-text-muted text-[10px] font-mono uppercase mt-1">
              Points
            </p>
          </div>
          <div>
            <p className="text-white text-2xl font-light font-mono">
              {metadata?.frames_processed ?? 0}
            </p>
            <p className="text-text-muted text-[10px] font-mono uppercase mt-1">
              Frames
            </p>
          </div>
          <div>
            <p className="text-white text-2xl font-light font-mono">
              {metadata?.faces_detected ?? 0}
            </p>
            <p className="text-text-muted text-[10px] font-mono uppercase mt-1">
              Faces
            </p>
          </div>
        </div>
      </div>

      <div className="flex gap-3">
        <a
          href={result.video_url}
          download
          className="btn-primary flex-1 text-center"
        >
          Download
        </a>
        <button
          onClick={onOpenTips}
          className="btn-secondary"
        >
          How to hide
        </button>
      </div>
    </div>
  );
}

function TrackabilityBadge({ score }: { score: number }) {
  const getLevel = (s: number) => {
    if (s >= 70) return { label: "High", color: "text-danger" };
    if (s >= 40) return { label: "Medium", color: "text-warning" };
    return { label: "Low", color: "text-success" };
  };

  const level = getLevel(score);

  return (
    <div className="flex items-center gap-2">
      <span className="text-white text-lg font-mono">{score}%</span>
      <span className={`text-[10px] font-mono uppercase ${level.color}`}>
        {level.label}
      </span>
    </div>
  );
}
