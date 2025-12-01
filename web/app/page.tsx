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
      <header className="px-6 py-6 flex items-center justify-between">
        <span className="text-lg font-semibold tracking-tight">
          <span className="text-accent">a</span>ftertrace
        </span>
        <span className="text-text-muted text-sm font-mono">v1.0</span>
      </header>

      {/* Main content */}
      <div className="flex-1 flex flex-col justify-center px-6 pb-12">
        <div className="w-full max-w-lg mx-auto">
          {/* Hero text */}
          <div className="mb-10 text-center md:text-left animate-fade-in">
            <h1 className="text-4xl md:text-5xl font-bold leading-tight tracking-tight">
              see what
              <br />
              <span className="text-accent">cameras see</span>
            </h1>
            <p className="text-text-secondary mt-4 text-lg max-w-md">
              transform your clips into surveillance art. understand your digital footprint.
            </p>
          </div>

          {/* Card */}
          <div className="card p-6 space-y-6 animate-slide-up" style={{ animationDelay: "0.1s" }}>
            <UploadZone
              onFileSelect={handleFileSelect}
              disabled={isLoading}
              error={validationError}
            />

            <div className="border-t border-white/5 pt-6">
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
