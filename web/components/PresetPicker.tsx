"use client";

type Preset = {
  id: string;
  name: string;
  description: string;
};

const PRESETS: Preset[] = [
  { id: "blob_track", name: "Blob Track", description: "Boxes + IDs + connections" },
  { id: "particle_silhouette", name: "Particle Cloud", description: "Ethereal point cloud" },
  { id: "number_cloud", name: "Number Cloud", description: "IDs on subject" },
  { id: "motion_trace", name: "Motion Trace", description: "Flowing motion lines" },
  { id: "contour_trace", name: "Contour", description: "Edge detection" },
  { id: "face_scanner", name: "Face Scanner", description: "Minimal detection" },
  { id: "biometric", name: "Biometric", description: "Full CCTV analysis" },
  { id: "grid_trace", name: "Grid", description: "Geometric network" },
  { id: "data_body", name: "Data Body", description: "Code silhouette" },
  { id: "heat_map", name: "Thermal", description: "Heat signature" },
  { id: "catodic_cube", name: "Catodic", description: "CRT depth glitch" },
  { id: "ember_trails", name: "Ember", description: "Spark trails" },
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
      <div className="flex items-center justify-between mb-4">
        <span className="text-text-muted text-[11px] font-mono uppercase tracking-widest">
          Effect
        </span>
        <span className="text-text-muted text-[11px] font-mono">
          {currentIndex + 1} / {PRESETS.length}
        </span>
      </div>
      
      {/* Current selection display */}
      <div className="mb-4 pb-4 border-b border-white/[0.06]">
        <p className="text-white text-sm font-medium">{current.name}</p>
        <p className="text-text-muted text-xs mt-0.5">{current.description}</p>
      </div>
      
      {/* Preset grid */}
      <div className="flex flex-wrap gap-1.5">
        {PRESETS.map((preset) => (
          <button
            key={preset.id}
            onClick={() => onChange(preset.id)}
            disabled={disabled}
            className={`
              px-2.5 py-1.5 text-[10px] font-mono uppercase tracking-wide
              border transition-all duration-150
              disabled:opacity-40 disabled:cursor-not-allowed
              ${value === preset.id 
                ? "border-white bg-white text-black" 
                : "border-white/10 text-text-muted hover:border-white/30 hover:text-text-secondary"
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
