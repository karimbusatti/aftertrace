"use client";

type Preset = {
  id: string;
  name: string;
  description: string;
};

const PRESETS: Preset[] = [
  { id: "blob_track", name: "Blob Track", description: "boxes with IDs and connections" },
  { id: "number_cloud", name: "Number Cloud", description: "subject becomes numbers" },
  { id: "particle_silhouette", name: "Particle Cloud", description: "point silhouette" },
  { id: "contour_trace", name: "Contour", description: "edge visualization" },
  { id: "face_scanner", name: "Face Scanner", description: "detection boxes" },
  { id: "biometric", name: "Biometric", description: "CCTV analysis" },
  { id: "motion_trace", name: "Motion Trace", description: "flowing trails" },
  { id: "grid_trace", name: "Grid", description: "geometric network" },
  { id: "data_body", name: "Data Body", description: "code silhouette" },
  { id: "heat_map", name: "Thermal", description: "heat signature" },
  { id: "catodic_cube", name: "Catodic", description: "CRT glitch" },
  { id: "ember_trails", name: "Ember", description: "spark trails" },
];

interface PresetPickerProps {
  value: string;
  onChange: (preset: string) => void;
  disabled?: boolean;
}

export function PresetPicker({ value, onChange, disabled }: PresetPickerProps) {
  return (
    <div>
      <p className="text-text-muted text-sm mb-3">choose effect</p>
      
      {/* Main presets - first row */}
      <div className="flex flex-wrap gap-2 mb-3">
        {PRESETS.slice(0, 4).map((preset) => (
          <button
            key={preset.id}
            onClick={() => onChange(preset.id)}
            disabled={disabled}
            className={`
              px-4 py-2.5 rounded-full text-sm font-medium
              transition-all duration-200
              disabled:opacity-50 disabled:cursor-not-allowed
              ${value === preset.id 
                ? "bg-accent text-white" 
                : "bg-white/5 text-text-secondary hover:bg-white/10 hover:text-white"
              }
            `}
          >
            {preset.name}
            <span className="text-xs ml-1 opacity-60">
              {preset.description}
            </span>
          </button>
        ))}
      </div>
      
      {/* More presets */}
      <div className="flex flex-wrap gap-2">
        {PRESETS.slice(4).map((preset) => (
          <button
            key={preset.id}
            onClick={() => onChange(preset.id)}
            disabled={disabled}
            className={`
              px-3 py-1.5 rounded-full text-xs font-medium
              transition-all duration-200
              disabled:opacity-50 disabled:cursor-not-allowed
              ${value === preset.id 
                ? "bg-accent text-white" 
                : "bg-white/5 text-text-muted hover:bg-white/10 hover:text-text-secondary"
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
