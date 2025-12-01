"use client";

interface Preset {
  id: string;
  name: string;
  description: string;
  category: "detection" | "visual" | "abstract";
}

const PRESETS: Preset[] = [
  // Detection presets first
  {
    id: "face_scanner",
    name: "Face Scanner",
    description: "CCTV-style detection",
    category: "detection",
  },
  {
    id: "biometric",
    name: "Biometric",
    description: "full identity analysis",
    category: "detection",
  },
  {
    id: "face_mesh",
    name: "Face Mesh",
    description: "468-point mapping",
    category: "detection",
  },
  {
    id: "surveillance_glow",
    name: "Surveillance",
    description: "cold tracking overlay",
    category: "detection",
  },
  // Visual presets
  {
    id: "data_body",
    name: "Data Body",
    description: "silhouette from code",
    category: "visual",
  },
  {
    id: "numeric_aura",
    name: "Numeric Aura",
    description: "binary edge detection",
    category: "visual",
  },
  {
    id: "catodic_cube",
    name: "Catodic",
    description: "CRT depth glitch",
    category: "visual",
  },
  {
    id: "heat_map",
    name: "Thermal",
    description: "heat signature scan",
    category: "visual",
  },
  // Abstract presets
  {
    id: "grid_trace",
    name: "Grid Trace",
    description: "geometric network",
    category: "abstract",
  },
  {
    id: "soft_blobs",
    name: "Soft Blobs",
    description: "organic flow",
    category: "abstract",
  },
  {
    id: "ember_trails",
    name: "Ember",
    description: "particle trails",
    category: "abstract",
  },
];

interface PresetPickerProps {
  value: string;
  onChange: (preset: string) => void;
  disabled?: boolean;
}

export function PresetPicker({ value, onChange, disabled }: PresetPickerProps) {
  const detectionPresets = PRESETS.filter(p => p.category === "detection");
  const visualPresets = PRESETS.filter(p => p.category === "visual");
  const abstractPresets = PRESETS.filter(p => p.category === "abstract");

  return (
    <div className="w-full space-y-4">
      <p className="text-text-secondary text-sm font-medium">
        choose your lens
      </p>
      
      {/* Detection presets - featured */}
      <div className="space-y-2">
        <p className="text-text-muted text-xs uppercase tracking-wider font-mono">
          detection
        </p>
        <div className="grid grid-cols-2 gap-2">
          {detectionPresets.map((preset) => (
            <PresetButton
              key={preset.id}
              preset={preset}
              isSelected={value === preset.id}
              onClick={() => onChange(preset.id)}
              disabled={disabled}
            />
          ))}
        </div>
      </div>

      {/* Visual presets */}
      <div className="space-y-2">
        <p className="text-text-muted text-xs uppercase tracking-wider font-mono">
          visual
        </p>
        <div className="grid grid-cols-2 gap-2">
          {visualPresets.map((preset) => (
            <PresetButton
              key={preset.id}
              preset={preset}
              isSelected={value === preset.id}
              onClick={() => onChange(preset.id)}
              disabled={disabled}
            />
          ))}
        </div>
      </div>

      {/* Abstract presets */}
      <div className="space-y-2">
        <p className="text-text-muted text-xs uppercase tracking-wider font-mono">
          abstract
        </p>
        <div className="grid grid-cols-3 gap-2">
          {abstractPresets.map((preset) => (
            <PresetButton
              key={preset.id}
              preset={preset}
              isSelected={value === preset.id}
              onClick={() => onChange(preset.id)}
              disabled={disabled}
              compact
            />
          ))}
        </div>
      </div>
    </div>
  );
}

function PresetButton({ 
  preset, 
  isSelected, 
  onClick, 
  disabled,
  compact = false,
}: { 
  preset: Preset; 
  isSelected: boolean; 
  onClick: () => void;
  disabled?: boolean;
  compact?: boolean;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className={`
        w-full ${compact ? 'p-2' : 'p-3'} rounded-xl text-left transition-all duration-200 ease-out
        border outline-none
        ${isSelected
          ? "bg-accent/15 border-accent/50 shadow-[0_0_20px_rgba(255,77,0,0.15)]"
          : "bg-surface-overlay/30 border-white/5 hover:border-white/15 hover:bg-surface-overlay/60"
        }
        ${disabled ? "opacity-50 cursor-not-allowed" : "cursor-pointer"}
      `}
    >
      <p className={`font-semibold ${compact ? 'text-xs' : 'text-sm'} ${isSelected ? "text-accent" : "text-text-primary"}`}>
        {preset.name}
      </p>
      {!compact && (
        <p className="text-xs text-text-muted mt-0.5 leading-tight font-mono">
          {preset.description}
        </p>
      )}
    </button>
  );
}
