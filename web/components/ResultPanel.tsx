"use client";

import type { ProcessResponse } from "@/lib/api";
import { getDownloadUrl } from "@/lib/api";

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
        <div className="w-12 h-12 border-2 border-accent/30 border-t-accent rounded-full animate-spin mx-auto mb-4" />
        <p className="text-white font-medium">analyzing frame by frame...</p>
        <p className="text-text-muted text-sm mt-2">
          this usually takes about 30 seconds
        </p>
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
        <div className="flex items-center justify-between mb-4">
          <span className="text-text-muted text-sm">analysis results</span>
          <TrackabilityBadge score={trackability} />
        </div>
        
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
