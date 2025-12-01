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
      <header className="px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="w-1 h-1 bg-white rounded-full" />
          <span className="text-[11px] font-medium tracking-[0.25em] uppercase text-white/90">
            Aftertrace
          </span>
        </div>
      </header>

      {/* Main */}
      <div className="flex-1 flex flex-col justify-center px-6 pb-16">
        <div className="w-full max-w-sm mx-auto">
          {/* Hero */}
          <div 
            className="mb-8 opacity-0 animate-fade-in"
            style={{ animationDelay: "0.05s", animationFillMode: "forwards" }}
          >
            <h1 className="text-2xl font-light text-white leading-tight tracking-tight">
              See what cameras
              <br />
              <span className="font-medium">see about you</span>
            </h1>
            <p className="text-text-secondary text-xs mt-4 leading-relaxed max-w-xs">
              Transform clips into motion visualizations and understand your digital footprint.
            </p>
          </div>

          {/* Card */}
          <div 
            className="card p-4 space-y-4 opacity-0 animate-slide-up"
            style={{ animationDelay: "0.1s", animationFillMode: "forwards" }}
          >
            <UploadZone
              onFileSelect={handleFileSelect}
              disabled={isLoading}
              error={validationError}
            />

            <div className="border-t border-white/[0.03] pt-4">
              <PresetPicker
                value={preset}
                onChange={setPreset}
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
                  <span className="flex items-center justify-center gap-2">
                    <span className="w-2.5 h-2.5 border border-black/20 border-t-black rounded-full animate-spin" />
                    <span>Analyzing</span>
                  </span>
                ) : (
                  "Create"
                )}
              </button>
              <p className="text-text-muted text-[9px] text-center mt-2.5 font-mono">
                ~30 seconds
              </p>
            </div>
          </div>

          {/* Results */}
          {(result || error || isLoading) && (
            <div 
              className="mt-5 opacity-0 animate-fade-in"
              style={{ animationDelay: "0.05s", animationFillMode: "forwards" }}
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
      <footer className="px-6 py-5 text-center">
        <p className="text-text-muted text-[9px] font-mono uppercase tracking-[0.2em] mb-1.5">
          No data stored
        </p>
        <p className="text-text-secondary text-[11px]">
          <a 
            href="https://instagram.com/thechildofvenus" 
            target="_blank" 
            rel="noopener noreferrer"
            className="hover:text-white transition-colors duration-150"
          >
            @thechildofvenus
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
