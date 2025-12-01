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
          relative h-28 flex flex-col items-center justify-center
          border border-dashed transition-all duration-150 cursor-pointer
          ${disabled ? "opacity-30 cursor-not-allowed" : ""}
          ${isDragging ? "border-white bg-white/[0.01]" : "border-white/10 hover:border-white/20"}
          ${error ? "border-danger/30" : ""}
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
            <p className="text-white text-[11px] font-mono truncate max-w-[220px]">
              {fileName}
            </p>
            <p className="text-text-muted text-[9px] mt-1">
              Tap to change
            </p>
          </div>
        ) : (
          <div className="text-center">
            <div className="text-text-muted text-lg mb-1">↑</div>
            <p className="text-text-secondary text-[11px]">
              Drop or tap
            </p>
            <p className="text-text-muted text-[9px] mt-0.5 font-mono">
              &lt;20s
            </p>
          </div>
        )}
      </div>
      
      {error && (
        <p className="text-danger text-[9px] mt-1.5 font-mono">
          {error}
        </p>
      )}
    </div>
  );
}
