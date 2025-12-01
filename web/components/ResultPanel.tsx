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
      <div className="card p-6 text-center">
        <div className="flex flex-col items-center gap-4">
          <div className="relative">
            <div className="w-10 h-10 border border-white/20 border-t-white rounded-full animate-spin" />
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="w-4 h-4 bg-white/10 rounded-full" />
            </div>
          </div>
          <div>
            <p className="text-white text-sm">Analyzing frame by frame</p>
            <p className="text-text-muted text-[10px] mt-1 font-mono">
              Processing · This takes ~30 seconds
            </p>
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="card p-5 border-danger/20">
        <div className="flex items-start gap-3">
          <div className="w-1.5 h-1.5 bg-danger rounded-full mt-1.5 shrink-0" />
          <div>
            <p className="text-white text-sm">{error}</p>
            <p className="text-text-muted text-[10px] mt-1">
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
    <div className="space-y-3">
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
      <div className="card p-4">
        <div className="flex items-center justify-between mb-4">
          <span className="text-text-muted text-[10px] font-mono uppercase tracking-[0.2em]">
            Analysis
          </span>
          <TrackabilityBadge score={trackability} />
        </div>
        
        <div className="grid grid-cols-3 gap-3 text-center">
          <StatBlock 
            value={metadata?.points_tracked ?? 0} 
            label="Points" 
          />
          <StatBlock 
            value={metadata?.frames_processed ?? 0} 
            label="Frames" 
          />
          <StatBlock 
            value={metadata?.faces_detected ?? 0} 
            label="Faces" 
          />
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
        <button
          onClick={onOpenTips}
          className="btn-secondary"
        >
          Reduce footprint
        </button>
      </div>
    </div>
  );
}

function StatBlock({ value, label }: { value: number; label: string }) {
  return (
    <div className="py-2">
      <p className="text-white text-xl font-light font-mono tracking-tight">
        {value.toLocaleString()}
      </p>
      <p className="text-text-muted text-[9px] font-mono uppercase tracking-[0.15em] mt-1">
        {label}
      </p>
    </div>
  );
}

function TrackabilityBadge({ score }: { score: number }) {
  const getLevel = (s: number) => {
    if (s >= 70) return { label: "High", color: "bg-danger", textColor: "text-danger" };
    if (s >= 40) return { label: "Medium", color: "bg-warning", textColor: "text-warning" };
    return { label: "Low", color: "bg-success", textColor: "text-success" };
  };

  const level = getLevel(score);

  return (
    <div className="flex items-center gap-2">
      <div className={`w-1.5 h-1.5 rounded-full ${level.color}`} />
      <span className="text-white text-sm font-mono">{score}%</span>
      <span className={`text-[9px] font-mono uppercase tracking-wider ${level.textColor}`}>
        {level.label}
      </span>
    </div>
  );
}
