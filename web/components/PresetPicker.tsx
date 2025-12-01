"use client";

interface Preset {
  id: string;
  name: string;
  description: string;
  featured?: boolean;
}

const PRESETS: Preset[] = [
  // Featured presets first
  {
    id: "blob_track",
    name: "Blob Track",
    description: "coordinate boxes",
    featured: true,
  },
  {
    id: "particle_silhouette",
    name: "Particle Cloud",
    description: "point silhouette",
    featured: true,
  },
  {
    id: "number_cloud",
    name: "Number Cloud",
    description: "scattered IDs",
    featured: true,
  },
  {
    id: "face_scanner",
    name: "Face Scanner",
    description: "detection boxes",
    featured: true,
  },
  // Other presets
  {
    id: "biometric",
    name: "Biometric",
    description: "full analysis",
  },
  {
    id: "face_mesh",
    name: "Face Mesh",
    description: "468 points",
  },
  {
    id: "data_body",
    name: "Data Body",
    description: "text silhouette",
  },
  {
    id: "grid_trace",
    name: "Grid Trace",
    description: "geometric net",
  },
  {
    id: "heat_map",
    name: "Thermal",
    description: "heat signature",
  },
  {
    id: "catodic_cube",
    name: "Catodic",
    description: "CRT glitch",
  },
  {
    id: "ember_trails",
    name: "Ember",
    description: "particle trails",
  },
  {
    id: "soft_blobs",
    name: "Soft Blobs",
    description: "organic flow",
  },
];

interface PresetPickerProps {
  value: string;
  onChange: (preset: string) => void;
  disabled?: boolean;
}

export function PresetPicker({ value, onChange, disabled }: PresetPickerProps) {
  const featured = PRESETS.filter(p => p.featured);
  const others = PRESETS.filter(p => !p.featured);

  return (
    <div className="w-full space-y-4">
      <p className="text-text-muted text-xs uppercase tracking-wider font-mono">
        choose effect
      </p>
      
      {/* Featured presets - larger */}
      <div className="grid grid-cols-2 gap-2">
        {featured.map((preset) => (
          <button
            key={preset.id}
            type="button"
            onClick={() => onChange(preset.id)}
            disabled={disabled}
            className={`
              w-full p-3 rounded-xl text-left transition-all duration-200 ease-out
              border outline-none
              ${value === preset.id
                ? "bg-white/10 border-white/40"
                : "bg-white/[0.03] border-white/[0.06] hover:border-white/20 hover:bg-white/[0.06]"
              }
              ${disabled ? "opacity-50 cursor-not-allowed" : "cursor-pointer"}
            `}
          >
            <p className={`font-medium text-sm ${value === preset.id ? "text-white" : "text-text-primary"}`}>
              {preset.name}
            </p>
            <p className="text-[11px] text-text-muted mt-0.5 font-mono">
              {preset.description}
            </p>
          </button>
        ))}
      </div>

      {/* Other presets - smaller grid */}
      <div className="grid grid-cols-4 gap-1.5">
        {others.map((preset) => (
          <button
            key={preset.id}
            type="button"
            onClick={() => onChange(preset.id)}
            disabled={disabled}
            className={`
              w-full py-2 px-2 rounded-lg text-center transition-all duration-200 ease-out
              border outline-none
              ${value === preset.id
                ? "bg-white/10 border-white/40"
                : "bg-white/[0.02] border-white/[0.04] hover:border-white/15 hover:bg-white/[0.05]"
              }
              ${disabled ? "opacity-50 cursor-not-allowed" : "cursor-pointer"}
            `}
          >
            <p className={`font-medium text-[11px] ${value === preset.id ? "text-white" : "text-text-secondary"}`}>
              {preset.name}
            </p>
          </button>
        ))}
      </div>
    </div>
  );
}
