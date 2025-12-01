"use client";

type Preset = {
  id: string;
  name: string;
  description: string;
};

const PRESETS: Preset[] = [
  { id: "blob_track", name: "Blob Track", description: "Boxes with IDs and connections" },
  { id: "number_cloud", name: "Number Cloud", description: "IDs on subject only" },
  { id: "particle_silhouette", name: "Particle Cloud", description: "Ethereal point silhouette" },
  { id: "face_scanner", name: "Face Scanner", description: "Minimal detection boxes" },
  { id: "biometric", name: "Biometric", description: "CCTV-style data overlay" },
  { id: "numeric_aura", name: "Numeric Aura", description: "Data-body contour" },
  { id: "motion_trace", name: "Motion Trace", description: "Clean optical flow trails" },
];

interface PresetPickerProps {
  value: string;
  onChange: (preset: string) => void;
  disabled?: boolean;
}

export function PresetPicker({ value, onChange, disabled }: PresetPickerProps) {
  return (
    <div>
      <div className="flex items-center justify-between mb-3">
        <span className="text-text-muted text-[11px] font-mono uppercase tracking-widest">
          Effect
        </span>
        <span className="text-text-muted text-[11px] font-mono">
          {PRESETS.findIndex(p => p.id === value) + 1}/{PRESETS.length}
        </span>
      </div>
      
      <div className="flex flex-wrap gap-2">
        {PRESETS.map((preset) => (
          <button
            key={preset.id}
            onClick={() => onChange(preset.id)}
            disabled={disabled}
            className={`
              px-3 py-2 text-[11px] font-mono uppercase tracking-wide
              border transition-all duration-200
              disabled:opacity-40 disabled:cursor-not-allowed
              ${value === preset.id 
                ? "border-white bg-white text-black" 
                : "border-white/20 text-text-secondary hover:border-white/40 hover:text-white"
              }
            `}
            title={preset.description}
          >
            {preset.name}
          </button>
        ))}
      </div>
    </div>
  );
}
