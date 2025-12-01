"use client";

type Preset = {
  id: string;
  name: string;
  description: string;
};

const MAIN_PRESETS: Preset[] = [
  { id: "blob_track", name: "Blob Track", description: "coordinate boxes" },
  { id: "particle_silhouette", name: "Particle Cloud", description: "point silhouette" },
  { id: "number_cloud", name: "Number Cloud", description: "scattered IDs" },
  { id: "face_scanner", name: "Face Scanner", description: "detection boxes" },
];

const SECONDARY_PRESETS: Preset[] = [
  { id: "biometric", name: "Biometric", description: "" },
  { id: "face_mesh", name: "Face Mesh", description: "" },
  { id: "data_body", name: "Data Body", description: "" },
  { id: "grid_trace", name: "Grid Trace", description: "" },
  { id: "heat_map", name: "Thermal", description: "" },
  { id: "catodic_cube", name: "Catodic", description: "" },
  { id: "ember_trails", name: "Ember", description: "" },
  { id: "soft_blobs", name: "Soft Blobs", description: "" },
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
      
      {/* Main 4 presets - 2x2 grid */}
      <div className="grid grid-cols-2 gap-3 mb-3">
        {MAIN_PRESETS.map((preset) => (
          <button
            key={preset.id}
            onClick={() => onChange(preset.id)}
            disabled={disabled}
            className={`
              p-4 rounded-xl text-left transition-all duration-200
              disabled:opacity-50 disabled:cursor-not-allowed
              ${value === preset.id
                ? "bg-white/10 border-2 border-white/30" 
                : "bg-white/5 border-2 border-transparent hover:bg-white/8"
              }
            `}
          >
            <span className="block text-white font-semibold">{preset.name}</span>
            <span className="block text-text-muted text-sm font-mono mt-1">
              {preset.description}
            </span>
          </button>
        ))}
      </div>
      
      {/* Secondary presets - 2 rows of 4 */}
      <div className="grid grid-cols-4 gap-2">
        {SECONDARY_PRESETS.map((preset) => (
          <button
            key={preset.id}
            onClick={() => onChange(preset.id)}
            disabled={disabled}
            className={`
              py-3 px-2 rounded-xl text-center text-sm transition-all duration-200
              disabled:opacity-50 disabled:cursor-not-allowed
              ${value === preset.id 
                ? "bg-white/10 border-2 border-white/30 text-white" 
                : "bg-white/5 border-2 border-transparent text-text-secondary hover:bg-white/8 hover:text-white"
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
