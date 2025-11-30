"use client";

import { useRef, useState, useCallback } from "react";

interface UploadZoneProps {
  onFileSelect: (file: File) => void;
  disabled?: boolean;
  error?: string | null;
}

export function UploadZone({ onFileSelect, disabled, error }: UploadZoneProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [fileName, setFileName] = useState<string | null>(null);

  const handleClick = () => {
    if (!disabled) {
      inputRef.current?.click();
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setFileName(file.name);
      onFileSelect(file);
    }
  };

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    if (!disabled) setIsDragging(true);
  }, [disabled]);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      if (disabled) return;

      const file = e.dataTransfer.files?.[0];
      if (file && file.type.startsWith("video/")) {
        setFileName(file.name);
        onFileSelect(file);
      }
    },
    [disabled, onFileSelect]
  );

  return (
    <div className="w-full">
      <input
        ref={inputRef}
        type="file"
        accept="video/*"
        capture="environment"
        onChange={handleChange}
        className="hidden"
        disabled={disabled}
      />

      <button
        type="button"
        onClick={handleClick}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        disabled={disabled}
        className={`
          w-full min-h-[200px] rounded-2xl border-2 border-dashed
          flex flex-col items-center justify-center gap-4 p-8
          transition-all duration-300 ease-out cursor-pointer
          ${isDragging
            ? "border-accent bg-accent/10 scale-[1.02]"
            : "border-white/10 bg-surface-overlay/50 hover:border-white/20 hover:bg-surface-overlay"
          }
          ${disabled ? "opacity-50 cursor-not-allowed" : ""}
        `}
      >
        {/* Icon */}
        <div className={`
          w-16 h-16 rounded-full flex items-center justify-center
          transition-all duration-300
          ${isDragging ? "bg-accent text-surface" : "bg-white/5 text-text-secondary"}
        `}>
          <svg
            width="28"
            height="28"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.5"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
            <polyline points="17 8 12 3 7 8" />
            <line x1="12" y1="3" x2="12" y2="15" />
          </svg>
        </div>

        {/* Text */}
        <div className="text-center">
          {fileName ? (
            <>
              <p className="font-display font-semibold text-lg text-text-primary mb-1">
                {fileName}
              </p>
              <p className="text-text-secondary text-sm">
                Tap to choose a different clip
              </p>
            </>
          ) : (
            <>
              <p className="font-display font-semibold text-lg text-text-primary mb-1">
                Record or upload a clip
              </p>
              <p className="text-text-secondary text-sm">
                Drag & drop or tap to browse
              </p>
            </>
          )}
          <p className="text-text-muted text-xs mt-2">
            your video is processed locally and deleted after download.
          </p>
        </div>
      </button>

      {/* Validation error message */}
      {error && (
        <div className="mt-4 p-4 rounded-xl bg-accent/10 border border-accent/20">
          <p className="text-accent text-sm font-medium mb-1">
            aftertrace likes short, sharp clips.
          </p>
          <p className="text-text-secondary text-sm">
            {error}
          </p>
        </div>
      )}
    </div>
  );
}

