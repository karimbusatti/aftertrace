"use client";

type Preset = {
  id: string;
  name: string;
  description: string;
};

const PRESETS: Preset[] = [
  { id: "blob_track", name: "Blob Track", description: "Clean boxes with IDs and connections" },
  { id: "number_cloud", name: "Number Cloud", description: "Subject becomes numbers, background visible" },
  { id: "particle_silhouette", name: "Particle Cloud", description: "Ethereal point silhouette" },
  { id: "contour_trace", name: "Contour", description: "Pure edge visualization" },
  { id: "face_scanner", name: "Face Scanner", description: "Minimal detection boxes" },
  { id: "biometric", name: "Biometric", description: "Full CCTV analysis mode" },
  { id: "motion_trace", name: "Motion Trace", description: "Elegant flowing motion trails" },
  { id: "grid_trace", name: "Grid", description: "Geometric network" },
  { id: "data_body", name: "Data Body", description: "Silhouette rebuilt from code" },
  { id: "heat_map", name: "Thermal", description: "Heat signature visualization" },
  { id: "catodic_cube", name: "Catodic", description: "CRT depth with RGB glitch" },
  { id: "ember_trails", name: "Ember", description: "Spark trails following motion" },
];

interface PresetPickerProps {
  value: string;
  onChange: (preset: string) => void;
  disabled?: boolean;
}

export function PresetPicker({ value, onChange, disabled }: PresetPickerProps) {
  const currentIndex = PRESETS.findIndex(p => p.id === value);
  const current = PRESETS[currentIndex] || PRESETS[0];

  return (
    <div>
      <div className="flex items-center justify-between mb-3">
        <span className="text-text-muted text-[10px] font-mono uppercase tracking-[0.2em]">
          Effect
        </span>
        <span className="text-text-muted text-[10px] font-mono">
          {currentIndex + 1} / {PRESETS.length}
        </span>
      </div>
      
      {/* Current selection */}
      <div className="mb-4 pb-3 border-b border-white/[0.04]">
        <p className="text-white text-sm">{current.name}</p>
        <p className="text-text-muted text-[11px] mt-0.5">{current.description}</p>
      </div>
      
      {/* Preset buttons */}
      <div className="flex flex-wrap gap-1">
        {PRESETS.map((preset) => (
          <button
            key={preset.id}
            onClick={() => onChange(preset.id)}
            disabled={disabled}
            className={`
              px-2 py-1 text-[9px] font-mono uppercase tracking-wide
              border transition-all duration-150
              disabled:opacity-35 disabled:cursor-not-allowed
              ${value === preset.id 
                ? "border-white bg-white text-black" 
                : "border-white/8 text-text-muted hover:border-white/20 hover:text-text-secondary"
              }
            `}
          >
            {preset.name}
          </button>
        ))}
      </div>
    </div>
  );
}
