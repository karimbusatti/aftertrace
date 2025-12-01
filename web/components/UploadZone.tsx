"use client";

import { useCallback, useRef, useState } from "react";

interface UploadZoneProps {
  onFileSelect: (file: File) => void;
  disabled?: boolean;
  error?: string | null;
}

export function UploadZone({ onFileSelect, disabled, error }: UploadZoneProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [fileName, setFileName] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      if (disabled) return;
      const file = e.dataTransfer.files[0];
      if (file && file.type.startsWith("video/")) {
        setFileName(file.name);
        onFileSelect(file);
      }
    },
    [disabled, onFileSelect]
  );

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

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

  return (
    <div>
      <div
        onClick={handleClick}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        className={`
          relative h-32 flex flex-col items-center justify-center
          border border-dashed transition-all duration-200 cursor-pointer
          ${disabled ? "opacity-35 cursor-not-allowed" : ""}
          ${isDragging ? "border-white bg-white/[0.02]" : "border-white/15 hover:border-white/25"}
          ${error ? "border-danger/40" : ""}
        `}
      >
        <input
          ref={inputRef}
          type="file"
          accept="video/*,video/mp4,video/quicktime,video/webm"
          onChange={handleChange}
          className="hidden"
          disabled={disabled}
        />
        
        {fileName ? (
          <div className="text-center px-4">
            <p className="text-white text-xs font-mono truncate max-w-[260px]">
              {fileName}
            </p>
            <p className="text-text-muted text-[10px] mt-1.5">
              Tap to change
            </p>
          </div>
        ) : (
          <div className="text-center">
            <div className="w-8 h-8 border border-white/20 flex items-center justify-center mx-auto mb-3">
              <svg 
                className="w-4 h-4 text-white/60" 
                fill="none" 
                viewBox="0 0 24 24" 
                stroke="currentColor"
              >
                <path 
                  strokeLinecap="round" 
                  strokeLinejoin="round" 
                  strokeWidth={1.5} 
                  d="M5 10l7-7m0 0l7 7m-7-7v18" 
                />
              </svg>
            </div>
            <p className="text-text-secondary text-xs">
              Drop video or tap to select
            </p>
            <p className="text-text-muted text-[10px] mt-1 font-mono">
              MP4 · MOV · WebM · Under 20s
            </p>
          </div>
        )}
      </div>
      
      {error && (
        <p className="text-danger text-[10px] mt-2 font-mono">
          {error}
        </p>
      )}
    </div>
  );
}
