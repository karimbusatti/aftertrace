"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import type { ProcessResponse } from "@/lib/api";
import { getDownloadUrl, getOriginalUrl } from "@/lib/api";
import { statsCopy, getTrackingSummary, getTrackingLevel, miscCopy } from "@/lib/copy";

interface ResultPanelProps {
  result: ProcessResponse | null;
  error: string | null;
  isLoading: boolean;
  onOpenTips?: () => void;
}

export function ResultPanel({ result, error, isLoading, onOpenTips }: ResultPanelProps) {
  if (isLoading) {
    return (
      <div className="w-full p-8 rounded-2xl bg-surface-overlay/50 border border-white/5">
        <div className="flex flex-col items-center gap-4">
          {/* Animated loader */}
          <div className="relative w-12 h-12">
            <div className="absolute inset-0 rounded-full border-2 border-white/10" />
            <div className="absolute inset-0 rounded-full border-2 border-accent border-t-transparent animate-spin" />
          </div>
          
          <div className="text-center">
            <p className="font-display font-semibold text-text-primary">
              Analysing your video…
            </p>
            <p className="text-text-secondary text-sm mt-1">
              {miscCopy.processingHint}
            </p>
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="w-full p-6 rounded-2xl bg-red-500/10 border border-red-500/20">
        <div className="flex items-start gap-3">
          <div className="w-8 h-8 rounded-full bg-red-500/20 flex items-center justify-center flex-shrink-0">
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
              <path
                d="M8 5v3M8 11h.01"
                stroke="#ef4444"
                strokeWidth="2"
                strokeLinecap="round"
              />
            </svg>
          </div>
          <div>
            <p className="font-semibold text-red-400">Something went wrong</p>
            <p className="text-red-400/80 text-sm mt-1">{error}</p>
          </div>
        </div>
      </div>
    );
  }

  if (!result) return null;

  const { metadata } = result;

  return (
    <div className="w-full space-y-4 animate-fade-in">
      {/* Video Preview Card */}
      <VideoPreview 
        jobId={result.job_id} 
        filename={result.filename}
        hasOriginal={!!result.original_download_url}
      />

      {/* Main result card */}
      <div className="p-6 rounded-2xl bg-surface-overlay/50 border border-white/5">
        {/* Success header */}
        <div className="flex items-center gap-3 mb-6">
          <div className="w-10 h-10 rounded-full bg-accent/20 flex items-center justify-center">
            <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
              <path
                d="M5 10L8.5 13.5L15 7"
                stroke="#FF6B35"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          </div>
          <div>
            <p className="font-display font-semibold text-text-primary">
              Your Aftertrace is ready
            </p>
            <p className="text-text-secondary text-sm">{result.filename}</p>
          </div>
        </div>

        {/* Stats grid with explainers */}
        <div className="grid grid-cols-2 gap-3">
          <StatCardWithExplainer
            statKey="frames_processed"
            value={metadata.frames_processed}
          />
          <StatCardWithExplainer
            statKey="total_points_spawned"
            value={metadata.total_points_spawned}
          />
          <StatCardWithExplainer
            statKey="average_points_per_frame"
            value={metadata.average_points_per_frame}
          />
          <StatCardWithExplainer
            statKey="beats_detected"
            value={metadata.beats_detected}
            note={metadata.beats_detected === 0 ? "no audio" : undefined}
          />
        </div>

        {/* Processing info */}
        <div className="mt-4 p-3 rounded-lg bg-surface-raised border border-white/5">
          <div className="flex justify-between text-sm">
            <span className="text-text-secondary">Processing time</span>
            <span className="text-text-primary font-mono">
              {metadata.processing_time_seconds.toFixed(1)}s
            </span>
          </div>
          <div className="flex justify-between text-sm mt-1">
            <span className="text-text-secondary">Duration</span>
            <span className="text-text-primary font-mono">
              {metadata.duration_seconds.toFixed(1)}s
            </span>
          </div>
        </div>

        {/* Download button */}
        <a
          href={getDownloadUrl(result.job_id)}
          download={`aftertrace_${result.job_id}.mp4`}
          className="btn-primary w-full mt-6 text-center block"
        >
          Download Video
        </a>

        {/* Privacy note */}
        <p className="text-text-muted text-xs text-center mt-4">
          {miscCopy.downloadNote}
        </p>
      </div>

      {/* Surveillance Insight card */}
      <SurveillanceInsight
        trackabilityScore={metadata.trackability_score}
        longestTrackSeconds={metadata.longest_track_seconds}
        maxTrackingFrames={metadata.max_continuous_tracking_frames}
        peopleDetected={metadata.people_detected}
        avgPointsPerFrame={metadata.average_points_per_frame}
        onOpenTips={onOpenTips}
      />
    </div>
  );
}

// =============================================================================
// VIDEO PREVIEW WITH ALTERNATING PLAYBACK
// =============================================================================

type PlaybackMode = "effect" | "alternate";

// Alternation timing (in seconds)
const ORIGINAL_DURATION = 1.5;  // Show original for 1.5s
const EFFECT_DURATION = 3.0;    // Show effect for 3s

function VideoPreview({ 
  jobId, 
  filename,
  hasOriginal,
}: { 
  jobId: string; 
  filename: string;
  hasOriginal: boolean;
}) {
  const [isPlaying, setIsPlaying] = useState(false);
  const [hasError, setHasError] = useState(false);
  const [playbackMode, setPlaybackMode] = useState<PlaybackMode>("effect");
  const [showingOriginal, setShowingOriginal] = useState(false);
  
  const effectVideoRef = useRef<HTMLVideoElement>(null);
  const originalVideoRef = useRef<HTMLVideoElement>(null);
  const alternateTimerRef = useRef<NodeJS.Timeout | null>(null);
  
  const effectUrl = getDownloadUrl(jobId);
  const originalUrl = getOriginalUrl(jobId);

  // Cleanup timer on unmount
  useEffect(() => {
    return () => {
      if (alternateTimerRef.current) {
        clearTimeout(alternateTimerRef.current);
      }
    };
  }, []);

  // Handle alternating playback
  const scheduleNextSwitch = useCallback(() => {
    if (playbackMode !== "alternate") return;
    
    const duration = showingOriginal ? ORIGINAL_DURATION : EFFECT_DURATION;
    
    alternateTimerRef.current = setTimeout(() => {
      setShowingOriginal(prev => !prev);
      scheduleNextSwitch();
    }, duration * 1000);
  }, [playbackMode, showingOriginal]);

  // Start/stop alternation when mode changes
  useEffect(() => {
    if (alternateTimerRef.current) {
      clearTimeout(alternateTimerRef.current);
      alternateTimerRef.current = null;
    }
    
    if (playbackMode === "alternate" && isPlaying) {
      scheduleNextSwitch();
    } else {
      setShowingOriginal(false);
    }
  }, [playbackMode, isPlaying, scheduleNextSwitch]);

  // Sync video playback times when switching
  useEffect(() => {
    if (playbackMode !== "alternate") return;
    
    const effectVideo = effectVideoRef.current;
    const originalVideo = originalVideoRef.current;
    
    if (effectVideo && originalVideo) {
      // Sync the hidden video to the visible one's time
      if (showingOriginal) {
        originalVideo.currentTime = effectVideo.currentTime;
      } else {
        effectVideo.currentTime = originalVideo.currentTime;
      }
    }
  }, [showingOriginal, playbackMode]);

  const handlePlay = () => {
    setIsPlaying(true);
    // Also start the original video if in alternate mode
    if (playbackMode === "alternate" && originalVideoRef.current) {
      originalVideoRef.current.play().catch(() => {});
    }
  };

  const handlePause = () => {
    setIsPlaying(false);
    // Also pause the original video
    if (originalVideoRef.current) {
      originalVideoRef.current.pause();
    }
  };

  if (hasError) {
    return (
      <div className="rounded-2xl overflow-hidden bg-surface-raised border border-white/5 p-8 text-center">
        <p className="text-text-secondary text-sm">Preview unavailable</p>
        <a
          href={effectUrl}
          download={`aftertrace_${jobId}.mp4`}
          className="text-accent text-sm hover:underline mt-2 inline-block"
        >
          Download instead →
        </a>
      </div>
    );
  }

  return (
    <div className="rounded-2xl overflow-hidden bg-black border border-white/10 shadow-2xl shadow-black/50">
      {/* Video container with aspect ratio */}
      <div className="relative aspect-video bg-surface">
        {/* Effect video (main) */}
        <video
          ref={effectVideoRef}
          src={effectUrl}
          controls
          autoPlay
          muted
          loop
          playsInline
          onPlay={handlePlay}
          onPause={handlePause}
          onError={() => setHasError(true)}
          className={`w-full h-full object-contain absolute inset-0 transition-opacity duration-300
            ${playbackMode === "alternate" && showingOriginal ? "opacity-0" : "opacity-100"}`}
        />
        
        {/* Original video (for alternating) - hidden but synced */}
        {hasOriginal && playbackMode === "alternate" && (
          <video
            ref={originalVideoRef}
            src={originalUrl}
            muted
            loop
            playsInline
            className={`w-full h-full object-contain absolute inset-0 transition-opacity duration-300
              ${showingOriginal ? "opacity-100" : "opacity-0"}`}
          />
        )}
        
        {/* Label showing which video is displayed */}
        {playbackMode === "alternate" && (
          <div className="absolute top-3 left-3 z-10">
            <span className={`
              px-2 py-1 rounded text-xs font-medium backdrop-blur-sm
              transition-all duration-300
              ${showingOriginal 
                ? "bg-white/20 text-white" 
                : "bg-accent/30 text-accent border border-accent/30"
              }
            `}>
              {showingOriginal ? "original" : "effect"}
            </span>
          </div>
        )}
        
        {/* Overlay gradient for polish */}
        <div className="absolute inset-0 pointer-events-none bg-gradient-to-t from-black/20 via-transparent to-transparent" />
      </div>
      
      {/* Video info bar */}
      <div className="px-4 py-3 bg-surface-raised/80 backdrop-blur">
        {/* Playback mode toggle - only show if original is available */}
        {hasOriginal && (
          <div className="flex items-center justify-center gap-1 mb-3">
            <button
              onClick={() => setPlaybackMode("effect")}
              className={`
                px-3 py-1.5 rounded-l-lg text-xs font-medium transition-all
                ${playbackMode === "effect"
                  ? "bg-accent text-white"
                  : "bg-white/5 text-text-secondary hover:bg-white/10"
                }
              `}
            >
              {miscCopy.effectOnlyLabel}
            </button>
            <button
              onClick={() => setPlaybackMode("alternate")}
              className={`
                px-3 py-1.5 rounded-r-lg text-xs font-medium transition-all
                ${playbackMode === "alternate"
                  ? "bg-accent text-white"
                  : "bg-white/5 text-text-secondary hover:bg-white/10"
                }
              `}
            >
              {miscCopy.alternateModeLabel}
            </button>
          </div>
        )}
        
        {/* Helper text for alternate mode */}
        {hasOriginal && playbackMode === "alternate" && (
          <p className="text-text-muted text-xs text-center mb-3 opacity-70">
            {miscCopy.alternateModeHelper}
          </p>
        )}
        
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            {isPlaying ? (
              <span className="flex items-center gap-1.5 text-accent text-xs font-medium">
                <span className="relative flex h-2 w-2">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-accent opacity-75" />
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-accent" />
                </span>
                playing
              </span>
            ) : (
              <span className="text-text-muted text-xs">paused</span>
            )}
          </div>
          
          <a
            href={effectUrl}
            download={`aftertrace_${jobId}.mp4`}
            className="text-text-secondary hover:text-accent text-xs font-medium transition-colors flex items-center gap-1"
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M7 10l5 5 5-5M12 15V3" />
            </svg>
            download
          </a>
        </div>
      </div>
    </div>
  );
}

// =============================================================================
// STAT CARD WITH EXPLAINER
// =============================================================================

type StatKey = keyof typeof statsCopy;

function StatCardWithExplainer({
  statKey,
  value,
  note,
}: {
  statKey: StatKey;
  value: number | string | undefined;
  note?: string;
}) {
  // Hide if value is undefined/null
  if (value === undefined || value === null) return null;

  const copy = statsCopy[statKey];

  return (
    <div className="p-4 rounded-xl bg-surface-raised border border-white/5 group">
      <p className="text-text-secondary text-xs uppercase tracking-wider mb-1">
        {copy.label}
      </p>
      <p className="font-display font-bold text-2xl text-text-primary">
        {typeof value === "number" ? value.toLocaleString() : value}
      </p>
      {note && (
        <p className="text-text-muted text-xs mt-0.5">{note}</p>
      )}
      {/* Explainer - shown on hover/focus or always visible on mobile */}
      <p className="text-text-muted text-xs mt-2 leading-relaxed opacity-60 group-hover:opacity-100 transition-opacity">
        {copy.explainer}
      </p>
    </div>
  );
}

// =============================================================================
// SURVEILLANCE INSIGHT
// =============================================================================

function SurveillanceInsight({
  trackabilityScore,
  longestTrackSeconds,
  maxTrackingFrames,
  peopleDetected,
  avgPointsPerFrame,
  onOpenTips,
}: {
  trackabilityScore: number;
  longestTrackSeconds: number;
  maxTrackingFrames: number;
  peopleDetected: number;
  avgPointsPerFrame: number;
  onOpenTips?: () => void;
}) {
  const level = getTrackingLevel(trackabilityScore);
  const summary = getTrackingSummary(trackabilityScore);

  // Color based on tracking level
  const levelColors = {
    low: { text: "text-green-400", bar: "bg-green-500", bg: "bg-green-500/10" },
    medium: { text: "text-yellow-400", bar: "bg-yellow-500", bg: "bg-yellow-500/10" },
    high: { text: "text-red-400", bar: "bg-red-500", bg: "bg-red-500/10" },
  };
  const colors = levelColors[level];

  return (
    <div className="p-5 rounded-2xl bg-surface-overlay/30 border border-white/5">
      {/* Header */}
      <div className="flex items-center gap-2 mb-4">
        <svg 
          width="16" 
          height="16" 
          viewBox="0 0 16 16" 
          fill="none"
          className="text-text-secondary"
        >
          <circle cx="8" cy="8" r="7" stroke="currentColor" strokeWidth="1.5" />
          <circle cx="8" cy="8" r="3" stroke="currentColor" strokeWidth="1.5" />
          <circle cx="8" cy="8" r="1" fill="currentColor" />
        </svg>
        <span className="text-text-secondary text-sm font-medium">
          Surveillance Insight
        </span>
        {/* Live indicator - pulses on high tracking */}
        <LiveIndicator level={level} />
      </div>

      {/* Trackability Score */}
      <div className="mb-4">
        <div className="flex items-baseline justify-between mb-2">
          <span className="text-text-secondary text-sm">
            {statsCopy.trackability_score.label}
          </span>
          <span className={`font-display font-bold text-xl ${colors.text}`}>
            {trackabilityScore}/100
          </span>
        </div>
        
        {/* Score bar */}
        <div className="h-2 bg-white/5 rounded-full overflow-hidden">
          <div 
            className={`h-full rounded-full transition-all duration-500 ${colors.bar}`}
            style={{ width: `${trackabilityScore}%` }}
          />
        </div>
        
        {/* Explainer */}
        <p className="text-text-muted text-xs mt-2 opacity-70">
          {statsCopy.trackability_score.explainer}
        </p>
      </div>

      {/* Summary sentence */}
      <div className={`p-3 rounded-lg ${colors.bg} mb-4`}>
        <p className={`text-sm font-medium ${colors.text}`}>
          {summary.headline}
        </p>
        <p className="text-text-secondary text-sm mt-1">
          {summary.body}
        </p>
      </div>

      {/* Detailed stats */}
      <div className="space-y-3 pt-3 border-t border-white/5">
        {/* Longest track */}
        {longestTrackSeconds > 0 && (
          <InsightRow
            label={statsCopy.longest_track_seconds.label}
            value={`${longestTrackSeconds.toFixed(1)}s (${maxTrackingFrames} frames)`}
            explainer={statsCopy.longest_track_seconds.explainer}
          />
        )}

        {/* Avg points per frame */}
        {avgPointsPerFrame > 0 && (
          <InsightRow
            label={statsCopy.average_points_per_frame.label}
            value={avgPointsPerFrame.toFixed(1)}
            explainer={statsCopy.average_points_per_frame.explainer}
          />
        )}
        
        {/* People detected */}
        {peopleDetected > 0 && (
          <InsightRow
            label={statsCopy.people_detected.label}
            value={`${peopleDetected}`}
            explainer={statsCopy.people_detected.explainer}
          />
        )}
      </div>

      {/* Tips link */}
      {onOpenTips && (
        <button
          onClick={onOpenTips}
          className="w-full mt-4 pt-4 border-t border-white/5 
                     text-accent text-sm hover:underline transition-colors
                     flex items-center justify-center gap-2"
        >
          <span>→</span>
          <span>how to be less trackable</span>
        </button>
      )}
    </div>
  );
}

function InsightRow({
  label,
  value,
  explainer,
}: {
  label: string;
  value: string;
  explainer: string;
}) {
  return (
    <div className="group">
      <div className="flex justify-between items-baseline">
        <span className="text-text-muted text-xs uppercase tracking-wider">
          {label}
        </span>
        <span className="text-text-primary font-mono text-sm">
          {value}
        </span>
      </div>
      <p className="text-text-muted text-xs mt-1 opacity-50 group-hover:opacity-80 transition-opacity">
        {explainer}
      </p>
    </div>
  );
}

// =============================================================================
// LIVE INDICATOR
// Shows tracking activity status with appropriate animation
// =============================================================================

type TrackingLevel = "low" | "medium" | "high";

function LiveIndicator({ level }: { level: TrackingLevel }) {
  if (level === "low") {
    // No indicator for low tracking - you're doing fine
    return null;
  }
  
  if (level === "medium") {
    // Static amber dot - gentle awareness
    return (
      <span className="ml-auto flex items-center gap-1.5">
        <span className="w-1.5 h-1.5 rounded-full bg-yellow-500/60" />
        <span className="text-yellow-500/60 text-xs font-medium">watching</span>
      </span>
    );
  }
  
  // High tracking - pulsing red "live" indicator
  return (
    <span className="ml-auto flex items-center gap-1.5">
      <span className="relative flex h-2 w-2">
        <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75" />
        <span className="relative inline-flex rounded-full h-2 w-2 bg-red-500" />
      </span>
      <span className="text-red-400 text-xs font-medium">live</span>
    </span>
  );
}
