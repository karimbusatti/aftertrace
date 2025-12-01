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
      {/* 
        Note: NOT using capture attribute so mobile browsers show 
        the chooser dialog (record OR gallery) instead of going 
        straight to camera 
      */}
      <input
        ref={inputRef}
        type="file"
        accept="video/*"
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
          w-full min-h-[180px] rounded-2xl border-2 border-dashed
          flex flex-col items-center justify-center gap-3 p-6
          transition-all duration-300 ease-out cursor-pointer
          ${isDragging
            ? "border-accent bg-accent/10 scale-[1.02]"
            : "border-white/10 bg-surface-overlay/30 hover:border-white/20 hover:bg-surface-overlay/60"
          }
          ${disabled ? "opacity-50 cursor-not-allowed" : ""}
        `}
      >
        {/* Icon */}
        <div className={`
          w-14 h-14 rounded-full flex items-center justify-center
          transition-all duration-300
          ${isDragging ? "bg-accent text-white" : "bg-white/5 text-text-secondary"}
        `}>
          {fileName ? (
            <svg
              width="24"
              height="24"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <path d="M5 12l5 5L20 7" />
            </svg>
          ) : (
            <svg
              width="24"
              height="24"
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
          )}
        </div>

        {/* Text */}
        <div className="text-center">
          {fileName ? (
            <>
              <p className="font-semibold text-base text-text-primary mb-1 truncate max-w-[200px]">
                {fileName}
              </p>
              <p className="text-text-muted text-xs">
                tap to change
              </p>
            </>
          ) : (
            <>
              <p className="font-semibold text-base text-text-primary mb-1">
                upload a clip
              </p>
              <p className="text-text-muted text-xs">
                drag & drop, tap to browse, or record
              </p>
            </>
          )}
        </div>
        
        {/* Privacy note */}
        <p className="text-text-muted text-[10px] mt-1 font-mono opacity-60">
          processed server-side, deleted after download
        </p>
      </button>

      {/* Validation error message */}
      {error && (
        <div className="mt-3 p-3 rounded-xl bg-danger/10 border border-danger/20">
          <p className="text-danger text-sm font-medium mb-0.5">
            clip too heavy
          </p>
          <p className="text-text-secondary text-xs">
            {error}
          </p>
        </div>
      )}
    </div>
  );
}
