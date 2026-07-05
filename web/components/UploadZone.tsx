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
  const [notice, setNotice] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Inspect the clip client-side and warn if it'll be slow to process.
  const inspectFile = useCallback((file: File) => {
    setNotice(null);
    try {
      const url = URL.createObjectURL(file);
      const v = document.createElement("video");
      v.preload = "metadata";
      v.onloadedmetadata = () => {
        const longest = Math.max(v.videoWidth, v.videoHeight);
        const dur = v.duration || 0;
        const sizeMB = file.size / (1024 * 1024);
        const bits: string[] = [];
        if (longest >= 2000) bits.push("4K");
        else if (longest >= 1400) bits.push("1080p+");
        if (dur >= 15) bits.push(`${Math.round(dur)}s`);
        if (bits.length) {
          setNotice(
            `${bits.join(" · ")} clip detected (${sizeMB.toFixed(0)}MB). Higher quality and longer videos look great but take longer to render.`
          );
        }
        URL.revokeObjectURL(url);
      };
      v.onerror = () => URL.revokeObjectURL(url);
      v.src = url;
    } catch {
      /* metadata probe is best-effort */
    }
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      if (disabled) return;
      const file = e.dataTransfer.files[0];
      if (file && file.type.startsWith("video/")) {
        setFileName(file.name);
        inspectFile(file);
        onFileSelect(file);
      }
    },
    [disabled, onFileSelect, inspectFile]
  );

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setFileName(file.name);
      inspectFile(file);
      onFileSelect(file);
    }
  };

  return (
    <div>
      <label
        htmlFor="video-upload-input"
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        aria-label="Upload a video clip: drag and drop, or pick from your camera roll"
        className={`
          relative h-40 flex flex-col items-center justify-center rounded-2xl
          border-2 border-dashed transition-all duration-300 cursor-pointer
          focus-within:border-accent focus-within:ring-2 focus-within:ring-accent/40
          ${disabled ? "opacity-50 cursor-not-allowed" : ""}
          ${isDragging ? "border-accent bg-accent/5" : "border-white/20 hover:border-white/40"}
          ${error ? "border-danger/50" : ""}
        `}
      >
        <input
          ref={inputRef}
          id="video-upload-input"
          type="file"
          accept="video/*"
          onChange={handleChange}
          className="sr-only"
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
              drag & drop, or pick from your camera roll
            </p>
            <p className="text-text-muted text-xs mt-2">
              up to 4K · processed server-side, deleted after download
            </p>
          </div>
        )}
      </label>

      {notice && !error && (
        <p className="text-warning/90 text-xs mt-2 flex items-start gap-1.5">
          <span aria-hidden>⏳</span>
          <span>{notice}</span>
        </p>
      )}

      {error && (
        <p className="text-danger text-sm mt-2">
          {error}
        </p>
      )}
    </div>
  );
}
