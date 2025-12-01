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
      {/* Header - minimal */}
      <header className="px-6 py-6 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-2 h-2 bg-white rounded-full" />
          <span className="text-sm font-medium tracking-wide uppercase">
            Aftertrace
          </span>
        </div>
        <span className="text-text-muted text-xs font-mono">
          v1.0
        </span>
      </header>

      {/* Main content */}
      <div className="flex-1 flex flex-col justify-center px-6 pb-12">
        <div className="w-full max-w-lg mx-auto">
          {/* Hero text - elegant */}
          <div className="mb-12 animate-fade-in">
            <p className="text-text-muted text-xs font-mono uppercase tracking-widest mb-4">
              visual analysis tool
            </p>
            <h1 className="text-4xl md:text-5xl font-light text-white leading-[1.1] tracking-tight">
              See what cameras
              <br />
              <span className="font-normal">see about you</span>
            </h1>
            <p className="text-text-secondary text-sm mt-6 max-w-md leading-relaxed">
              Transform your clips into data visualizations. 
              Understand your digital footprint through motion tracking, 
              facial detection, and biometric analysis.
            </p>
          </div>

          {/* Card */}
          <div 
            className="card p-6 space-y-6 animate-slide-up"
            style={{ animationDelay: "0.15s" }}
          >
            <UploadZone
              onFileSelect={handleFileSelect}
              disabled={isLoading}
              error={validationError}
            />

            <div className="border-t border-white/[0.06] pt-6">
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
                    <span className="w-3 h-3 border border-black/30 border-t-black rounded-full animate-spin" />
                    Analyzing
                  </span>
                ) : (
                  "Create Aftertrace"
                )}
              </button>
              <p className="text-text-muted text-[11px] text-center mt-4 font-mono">
                Frame-by-frame analysis · ~30 seconds
              </p>
            </div>
          </div>

          {/* Results */}
          {(result || error || isLoading) && (
            <div className="mt-8">
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

      {/* Footer - minimal */}
      <footer className="px-6 py-8 text-center">
        <div className="space-y-2">
          <p className="text-text-muted text-[11px] font-mono uppercase tracking-widest">
            No data stored · Deleted after download
          </p>
          <p className="text-text-secondary text-xs">
            Made by{" "}
            <a 
              href="https://instagram.com/thechildofvenus" 
              target="_blank" 
              rel="noopener noreferrer"
              className="text-white hover:opacity-70 transition-opacity"
            >
              Karim
            </a>
          </p>
        </div>
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
