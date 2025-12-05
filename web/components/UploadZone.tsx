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
          relative h-40 flex flex-col items-center justify-center rounded-2xl
          border-2 border-dashed transition-all duration-300 cursor-pointer
          ${disabled ? "opacity-50 cursor-not-allowed" : ""}
          ${isDragging ? "border-accent bg-accent/5" : "border-white/20 hover:border-white/40"}
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
            <div className="w-12 h-12 rounded-full bg-accent/10 flex items-center justify-center mx-auto mb-3">
              <span className="text-accent text-xl">✓</span>
            </div>
            <p className="text-white font-medium truncate max-w-[250px]">
                {fileName}
              </p>
            <p className="text-text-muted text-sm mt-1">
              tap to change
              </p>
          </div>
          ) : (
          <div className="text-center">
            <div className="w-12 h-12 rounded-full bg-white/5 flex items-center justify-center mx-auto mb-3">
              <span className="text-2xl">↑</span>
            </div>
            <p className="text-white font-medium">
              upload a clip
              </p>
            <p className="text-text-secondary text-sm mt-1">
              drag & drop, tap to browse, or record
              </p>
          <p className="text-text-muted text-xs mt-2">
              processed server-side, deleted after download
          </p>
        </div>
        )}
      </div>

      {error && (
        <p className="text-danger text-sm mt-2">
            {error}
          </p>
      )}
    </div>
  );
}
