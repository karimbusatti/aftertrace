"use client";

interface Preset {
  id: string;
  name: string;
  description: string;
}

const PRESETS: Preset[] = [
  {
    id: "grid_trace",
    name: "Grid Trace",
    description: "Sharp geometric network",
  },
  {
    id: "data_body",
    name: "Data Body",
    description: "Silhouette from letters",
  },
  {
    id: "numeric_aura",
    name: "Numeric Aura",
    description: "Glowing 0s and 1s",
  },
  {
    id: "catodic_cube",
    name: "Catodic Cube",
    description: "3D depth with RGB glitch",
  },
  {
    id: "face_scanner",
    name: "Face Scanner",
    description: "AI face detection boxes",
  },
  {
    id: "face_mesh",
    name: "Face Mesh",
    description: "468-point face mapping",
  },
  {
    id: "surveillance_glow",
    name: "Surveillance Glow",
    description: "Cold tracking overlay",
  },
  {
    id: "biometric",
    name: "Biometric",
    description: "Full biometric analysis",
  },
  {
    id: "heat_map",
    name: "Heat Map",
    description: "Thermal vision tracking",
  },
  {
    id: "soft_blobs",
    name: "Soft Blobs",
    description: "Dreamy organic shapes",
  },
  {
    id: "ember_trails",
    name: "Ember Trails",
    description: "Sparks tracing movement",
  },
];

interface PresetPickerProps {
  value: string;
  onChange: (preset: string) => void;
  disabled?: boolean;
}

export function PresetPicker({ value, onChange, disabled }: PresetPickerProps) {
  return (
    <div className="w-full">
      <p className="text-text-secondary text-sm mb-3 font-medium">
        Choose your visual
      </p>
      
      <div className="grid grid-cols-2 gap-3">
        {PRESETS.map((preset) => (
          <button
            key={preset.id}
            type="button"
            onClick={() => onChange(preset.id)}
            disabled={disabled}
            className={`
              w-full p-3 rounded-xl text-left transition-all duration-200 ease-out
              border outline-none
              ${value === preset.id
                ? "bg-accent/10 border-accent"
                : "bg-surface-overlay/50 border-white/5 hover:border-white/15 hover:bg-surface-overlay hover:scale-[1.02] hover:-translate-y-0.5 focus-visible:border-white/20 focus-visible:ring-1 focus-visible:ring-white/10"
              }
              ${disabled ? "opacity-50 cursor-not-allowed" : "cursor-pointer"}
            `}
          >
            <p className={`font-display font-semibold text-sm ${value === preset.id ? "text-accent" : "text-text-primary"}`}>
              {preset.name}
            </p>
            <p className="text-xs text-text-secondary mt-0.5 leading-tight">
              {preset.description}
            </p>
          </button>
        ))}
      </div>
    </div>
  );
}
