"use client";

import { useState } from "react";
import { UploadZone } from "@/components/UploadZone";
import { PresetPicker } from "@/components/PresetPicker";
import { ResultPanel } from "@/components/ResultPanel";
import { TipsSheet } from "@/components/TipsSheet";
import { processVideo, type ProcessResponse } from "@/lib/api";

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [preset, setPreset] = useState("blob_track");
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<ProcessResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [validationError, setValidationError] = useState<string | null>(null);
  const [isTipsOpen, setIsTipsOpen] = useState(false);
  
  // Sequence mode state
  const [mode, setMode] = useState<'single' | 'sequence'>('single');
  const [sequence, setSequence] = useState<string[]>([]);
  const [maxSlots, setMaxSlots] = useState(3); // Default 3 slots, can choose 2-5
  const [segmentDuration, setSegmentDuration] = useState(0.5); // Default 0.5s per segment

  const handleSubmit = async () => {
    if (!file) return;

    setIsLoading(true);
    setError(null);
    setValidationError(null);
    setResult(null);

    try {
      // Build sequence config if in sequence mode
      const sequenceConfig = mode === 'sequence' ? {
        effects: sequence,
        segmentDuration: segmentDuration
      } : null;
      
      const response = await processVideo(file, preset, false, sequenceConfig);
      setResult(response);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Something went wrong";
      if (message.includes("too heavy") || message.includes("under 20 seconds")) {
        setValidationError(message);
      } else {
        setError(message);
      }
    } finally {
      setIsLoading(false);
    }
  };

  const handleFileSelect = (selectedFile: File) => {
    setFile(selectedFile);
    setValidationError(null);
  };

  const canSubmit = file && !isLoading && (mode === 'single' || sequence.length > 0);

  return (
    <main className="min-h-dvh flex flex-col">
      {/* Header */}
      <header className="px-6 py-6">
        <span className="text-lg font-semibold tracking-tight">
          <span className="text-accent">a</span>ftertrace
        </span>
      </header>

      {/* Main content */}
      <div className="flex-1 flex flex-col justify-center px-6 pb-12 pt-4 md:pt-0">
        <div className="w-full max-w-lg mx-auto">
          {/* Hero text */}
          <div className="mb-8 md:mb-10 text-center md:text-left animate-fade-in">
            <h1 className="text-4xl md:text-5xl font-bold leading-tight tracking-tight">
              see what
              <br />
              <span className="text-accent">cameras see</span>
            </h1>
            <p className="text-text-secondary mt-3 md:mt-4 text-lg max-w-md mx-auto md:mx-0 leading-relaxed">
              transform your clips into surveillance art. understand your digital footprint.
            </p>
          </div>

          {/* Card with glowing orange border */}
          <div className="card-glow p-5 md:p-6 space-y-5 md:space-y-6 animate-slide-up" style={{ animationDelay: "0.1s" }}>
            <UploadZone
              onFileSelect={handleFileSelect}
              disabled={isLoading}
              error={validationError}
            />

            <div className="border-t border-white/5 pt-6">
              <PresetPicker
                value={preset}
                onChange={setPreset}
                mode={mode}
                onModeChange={setMode}
                sequence={sequence}
                onSequenceChange={setSequence}
                maxSlots={maxSlots}
                onMaxSlotsChange={setMaxSlots}
                segmentDuration={segmentDuration}
                onSegmentDurationChange={setSegmentDuration}
                disabled={isLoading}
              />
            </div>

            <div className="pt-2">
              <button
                onClick={handleSubmit}
                disabled={!canSubmit}
                className="btn-primary w-full"
              >
                {isLoading ? (
                  <span className="flex items-center justify-center gap-3">
                    <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                    analyzing...
                  </span>
                ) : (
                  "create aftertrace"
                )}
              </button>
              <p className="text-text-muted text-xs text-center mt-3">
                frame-by-frame analysis. takes a moment.
              </p>
            </div>
          </div>

          {/* Results */}
          {(result || error || isLoading) && (
            <div className="mt-8 animate-fade-in">
              <ResultPanel
                result={result}
                error={error}
                isLoading={isLoading}
                onOpenTips={() => setIsTipsOpen(true)}
              />
            </div>
          )}
        </div>
      </div>

      {/* Footer */}
      <footer className="px-6 py-8 text-center space-y-3">
        <p className="text-text-muted text-xs font-mono">
          nothing stored · nothing tracked · your data stays yours
        </p>
        <p className="text-text-secondary text-sm">
          made by{" "}
          <a 
            href="https://instagram.com/thechildofvenus" 
            target="_blank" 
            rel="noopener noreferrer"
            className="text-accent hover:text-accent-soft transition-colors"
          >
            Karim
          </a>
        </p>
      </footer>

      <TipsSheet 
        isOpen={isTipsOpen} 
        onClose={() => setIsTipsOpen(false)} 
        trackabilityScore={result?.metadata?.trackability_score}
      />
    </main>
  );
}
