"use client";

import { useState, useRef, useCallback } from "react";
import { UploadZone } from "@/components/UploadZone";
import { PresetPicker } from "@/components/PresetPicker";
import { ResultPanel } from "@/components/ResultPanel";
import { TipsSheet } from "@/components/TipsSheet";
import { processVideo, type ProcessResponse } from "@/lib/api";

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [preset, setPreset] = useState("grid_trace");
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<ProcessResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [validationError, setValidationError] = useState<string | null>(null);
  const [isTipsOpen, setIsTipsOpen] = useState(false);
  
  // Parallax state for desktop card effect
  const cardRef = useRef<HTMLDivElement>(null);
  const [cardTransform, setCardTransform] = useState({ x: 0, y: 0 });
  
  const handleCardMouseMove = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    if (!cardRef.current || window.innerWidth < 768) return;
    
    const rect = cardRef.current.getBoundingClientRect();
    const centerX = rect.left + rect.width / 2;
    const centerY = rect.top + rect.height / 2;
    
    // Calculate offset from center (-1 to 1)
    const offsetX = (e.clientX - centerX) / (rect.width / 2);
    const offsetY = (e.clientY - centerY) / (rect.height / 2);
    
    // Very subtle rotation (max 2 degrees) and translate (max 3px)
    setCardTransform({
      x: offsetX * 2,
      y: offsetY * 2,
    });
  }, []);
  
  const handleCardMouseLeave = useCallback(() => {
    setCardTransform({ x: 0, y: 0 });
  }, []);

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
      // Check if this is a validation error (too heavy, etc.)
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
    setValidationError(null); // Clear validation error when new file selected
  };

  const canSubmit = file && !isLoading;

  return (
    <main className="min-h-dvh flex flex-col">
      {/* Header */}
      <header className="px-6 py-5">
        <span className="font-display font-bold text-lg tracking-tight">
          <span className="text-accent">a</span>
          <span className="text-text-primary">ftertrace</span>
        </span>
      </header>

      {/* Main content */}
      <div className="flex-1 flex flex-col justify-center px-6 pb-8">
        <div className="w-full max-w-md mx-auto">
          {/* Hero text */}
          <div className="mb-10 animate-fade-in">
            <h1 className="font-display font-bold text-display-md text-text-primary mb-4">
              See what cameras
              <br />
              <span className="text-accent">see about you</span>
            </h1>
            <p className="text-text-secondary text-lg leading-relaxed">
              Turn your clips into glitch-art visuals. See what's visible — and what you might want to hide.
            </p>
          </div>

          {/* Card */}
          <div 
            ref={cardRef}
            onMouseMove={handleCardMouseMove}
            onMouseLeave={handleCardMouseLeave}
            className="card p-6 space-y-6 animate-slide-up shadow-lg shadow-black/20 border border-white/10 parallax-card"
            style={{ 
              animationDelay: "0.1s",
              transform: `perspective(1000px) rotateY(${cardTransform.x}deg) rotateX(${-cardTransform.y}deg) translateZ(0)`,
            }}
          >
            <UploadZone
              onFileSelect={handleFileSelect}
              disabled={isLoading}
              error={validationError}
            />

            <PresetPicker
              value={preset}
              onChange={setPreset}
              disabled={isLoading}
            />

            <div className="group">
              <button
                onClick={handleSubmit}
                disabled={!canSubmit}
                className="btn-primary w-full"
              >
                {isLoading ? "Processing…" : "Create Aftertrace"}
              </button>
              <p className="text-text-muted text-xs text-center mt-2 opacity-0 group-hover:opacity-100 sm:opacity-0 max-sm:opacity-60 transition-opacity duration-200">
                this might take a few seconds — we're crunching every frame.
              </p>
            </div>
          </div>

          {/* Results */}
          {(result || error || isLoading) && (
            <div className="mt-6">
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
        <p className="text-text-muted text-sm">
          nothing stored. nothing tracked.
        </p>
      </footer>

      {/* Tips Sheet */}
      <TipsSheet 
        isOpen={isTipsOpen} 
        onClose={() => setIsTipsOpen(false)} 
      />
    </main>
  );
}
