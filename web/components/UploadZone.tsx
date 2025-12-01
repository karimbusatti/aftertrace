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
          relative h-36 flex flex-col items-center justify-center
          border border-dashed transition-all duration-300 cursor-pointer
          ${disabled ? "opacity-40 cursor-not-allowed" : ""}
          ${isDragging ? "border-white bg-white/5" : "border-white/20 hover:border-white/40"}
          ${error ? "border-danger/50" : ""}
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
            <p className="text-white text-sm font-mono truncate max-w-[280px]">
              {fileName}
            </p>
            <p className="text-text-muted text-[11px] mt-1">
              Tap to change
            </p>
          </div>
        ) : (
          <div className="text-center">
            <div className="flex items-center justify-center gap-2 mb-2">
              <div className="w-5 h-5 border border-white/30 flex items-center justify-center">
                <span className="text-xs">↑</span>
              </div>
            </div>
            <p className="text-text-secondary text-sm">
              Drop video or tap to select
            </p>
            <p className="text-text-muted text-[11px] mt-1">
              MP4, MOV, WebM · Under 20s
            </p>
          </div>
        )}
      </div>
      
      {error && (
        <p className="text-danger text-[11px] mt-2 font-mono">
          {error}
        </p>
      )}
    </div>
  );
}
