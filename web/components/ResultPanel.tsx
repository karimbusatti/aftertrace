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
      <div className="card p-5 text-center">
        <div className="flex flex-col items-center gap-3">
          <div className="w-8 h-8 border border-white/15 border-t-white rounded-full animate-spin" />
          <div>
            <p className="text-white text-xs">Processing</p>
            <p className="text-text-muted text-[9px] mt-0.5 font-mono">
              Frame by frame analysis
            </p>
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="card p-4 border-danger/15">
        <div className="flex items-start gap-2.5">
          <div className="w-1 h-1 bg-danger rounded-full mt-1.5 shrink-0" />
          <div>
            <p className="text-white text-xs">{error}</p>
            <p className="text-text-muted text-[9px] mt-0.5">
              Try a different video
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
    <div className="space-y-2.5">
      {/* Video */}
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
      
      {/* Stats */}
      <div className="card p-3.5">
        <div className="flex items-center justify-between mb-3">
          <span className="text-text-muted text-[9px] font-mono uppercase tracking-[0.2em]">
            Analysis
          </span>
          <TrackabilityBadge score={trackability} />
          </div>
        
        <div className="grid grid-cols-3 gap-2 text-center">
          <StatBlock value={metadata?.points_tracked ?? 0} label="Pts" />
          <StatBlock value={metadata?.frames_processed ?? 0} label="Frm" />
          <StatBlock value={metadata?.faces_detected ?? 0} label="Fce" />
        </div>
          </div>
          
      {/* Actions */}
      <div className="flex gap-2">
        <a
          href={result.video_url}
          download
          className="btn-primary flex-1 text-center"
        >
          Download
        </a>
        <button onClick={onOpenTips} className="btn-secondary">
          Tips
        </button>
      </div>
    </div>
  );
}

function StatBlock({ value, label }: { value: number; label: string }) {
  return (
    <div className="py-1">
      <p className="text-white text-lg font-light font-mono">
        {value.toLocaleString()}
      </p>
      <p className="text-text-muted text-[8px] font-mono uppercase tracking-wider mt-0.5">
        {label}
      </p>
    </div>
  );
}

function TrackabilityBadge({ score }: { score: number }) {
  const getLevel = (s: number) => {
    if (s >= 70) return { label: "High", color: "bg-danger" };
    if (s >= 40) return { label: "Med", color: "bg-warning" };
    return { label: "Low", color: "bg-success" };
  };

  const level = getLevel(score);

  return (
    <div className="flex items-center gap-1.5">
      <div className={`w-1 h-1 rounded-full ${level.color}`} />
      <span className="text-white text-xs font-mono">{score}%</span>
    </div>
  );
}
