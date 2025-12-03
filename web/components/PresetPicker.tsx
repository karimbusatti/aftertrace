"use client";

type Preset = {
  id: string;
  name: string;
  description: string;
};

const MAIN_PRESETS: Preset[] = [
  { id: "blob_track", name: "Blob Track", description: "coordinate tracking" },
  { id: "particle_silhouette", name: "Particle Cloud", description: "ethereal silhouette" },
  { id: "numeric_aura", name: "Numeric Aura", description: "blue number cloud" },
  { id: "thermal_scan", name: "Thermal Scan", description: "heat vision" },
];

const SECONDARY_PRESETS: Preset[] = [
  { id: "data_body", name: "Matrix Mode", description: "" },
  { id: "face_mesh", name: "Face Mesh", description: "" },
  { id: "face_scanner", name: "Face Scan", description: "" },
  { id: "grid_trace", name: "Grid Trace", description: "" },
  { id: "contour_trace", name: "Contour", description: "" },
];

interface PresetPickerProps {
  value: string;
  onChange: (preset: string) => void;
  disabled?: boolean;
}

export function PresetPicker({ value, onChange, disabled }: PresetPickerProps) {
  return (
    <div>
      <p className="text-text-muted text-xs font-mono uppercase tracking-widest mb-4">
        Choose Effect
      </p>
      
      {/* Main 4 presets - 2x2 grid, compact */}
      <div className="grid grid-cols-2 gap-2 mb-2">
        {MAIN_PRESETS.map((preset) => (
          <button
            key={preset.id}
            onClick={() => onChange(preset.id)}
            disabled={disabled}
            className={`
              py-2.5 px-3 rounded-lg text-left transition-all duration-200
              disabled:opacity-50 disabled:cursor-not-allowed
              ${value === preset.id
                ? "bg-white/10 border border-white/30" 
                : "bg-white/5 border border-transparent hover:bg-white/8"
              }
            `}
          >
            <span className="block text-white text-sm font-medium">{preset.name}</span>
            <span className="block text-text-muted text-xs font-mono">
              {preset.description}
            </span>
          </button>
        ))}
      </div>
      
      {/* Secondary presets - flexible row */}
      <div className="flex flex-wrap gap-1.5 justify-center">
        {SECONDARY_PRESETS.map((preset) => (
          <button
            key={preset.id}
            onClick={() => onChange(preset.id)}
            disabled={disabled}
            className={`
              py-2 px-3 rounded-lg text-center text-xs transition-all duration-200
              disabled:opacity-50 disabled:cursor-not-allowed flex-1 min-w-[70px] max-w-[100px]
              ${value === preset.id 
                ? "bg-white/10 border border-white/30 text-white" 
                : "bg-white/5 border border-transparent text-text-secondary hover:bg-white/8 hover:text-white"
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
