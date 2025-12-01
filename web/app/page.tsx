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

  const handleSubmit = async () => {
    if (!file) return;

    setIsLoading(true);
    setError(null);
    setValidationError(null);
    setResult(null);

    try {
      const response = await processVideo(file, preset);
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

  const canSubmit = file && !isLoading;

  return (
    <main className="min-h-dvh flex flex-col">
      {/* Header */}
      <header className="px-6 py-5 flex items-center justify-between">
        <div className="flex items-center gap-2.5">
          <div className="w-1.5 h-1.5 bg-white rounded-full animate-pulse" />
          <span className="text-xs font-medium tracking-[0.2em] uppercase">
            Aftertrace
          </span>
        </div>
        <span className="text-text-muted text-[10px] font-mono tracking-wider">
          v1.0
        </span>
      </header>

      {/* Main content */}
      <div className="flex-1 flex flex-col justify-center px-6 pb-12">
        <div className="w-full max-w-md mx-auto">
          {/* Hero */}
          <div className="mb-10 opacity-0 animate-fade-in" style={{ animationDelay: "0.1s", animationFillMode: "forwards" }}>
            <p className="text-text-muted text-[10px] font-mono uppercase tracking-[0.3em] mb-3">
              Visual Analysis
            </p>
            <h1 className="text-3xl md:text-4xl font-light text-white leading-[1.15] tracking-tight">
              See what cameras
              <br />
              <span className="font-medium">see about you</span>
            </h1>
            <p className="text-text-secondary text-sm mt-5 leading-relaxed max-w-sm">
              Transform clips into motion visualizations. 
              Understand your digital footprint through tracking analysis.
            </p>
          </div>

          {/* Card */}
          <div 
            className="card p-5 space-y-5 opacity-0 animate-slide-up"
            style={{ animationDelay: "0.2s", animationFillMode: "forwards" }}
          >
            <UploadZone
              onFileSelect={handleFileSelect}
              disabled={isLoading}
              error={validationError}
            />

            <div className="border-t border-white/[0.04] pt-5">
              <PresetPicker
                value={preset}
                onChange={setPreset}
                disabled={isLoading}
              />
            </div>

            <div className="pt-3">
              <button
                onClick={handleSubmit}
                disabled={!canSubmit}
                className="btn-primary w-full relative overflow-hidden"
              >
                {isLoading ? (
                  <span className="flex items-center justify-center gap-2.5">
                    <span className="w-3 h-3 border border-black/20 border-t-black rounded-full animate-spin" />
                    <span>Analyzing</span>
                  </span>
                ) : (
                  "Create Aftertrace"
                )}
              </button>
              <p className="text-text-muted text-[10px] text-center mt-3 font-mono tracking-wide">
                Frame-by-frame · ~30 sec
              </p>
            </div>
          </div>

          {/* Results */}
          {(result || error || isLoading) && (
            <div 
              className="mt-6 opacity-0 animate-fade-in"
              style={{ animationDelay: "0.1s", animationFillMode: "forwards" }}
            >
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
      <footer className="px-6 py-6 text-center">
        <p className="text-text-muted text-[10px] font-mono uppercase tracking-[0.2em] mb-2">
          No data stored · Deleted after download
        </p>
        <p className="text-text-secondary text-xs">
          Made by{" "}
          <a 
            href="https://instagram.com/thechildofvenus" 
            target="_blank" 
            rel="noopener noreferrer"
            className="text-white hover:opacity-60 transition-opacity duration-200"
          >
            Karim
          </a>
        </p>
      </footer>

      {/* Tips Sheet */}
      <TipsSheet 
        isOpen={isTipsOpen} 
        onClose={() => setIsTipsOpen(false)} 
        trackabilityScore={result?.metadata?.trackability_score}
      />
    </main>
  );
}
