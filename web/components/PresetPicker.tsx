"use client";

import { SequenceTimeline } from "./SequenceTimeline";

type Preset = {
  id: string;
  name: string;
  description: string;
};

const MAIN_PRESETS: Preset[] = [
  { id: "codenet_overlay", name: "CodeNet", description: "feature network" },
  { id: "code_shadow", name: "CodeShadow", description: "ascii matrix" },
  { id: "binary_bloom", name: "Binary Bloom", description: "0/1 silhouette" },
  { id: "blob_track", name: "Blob Track", description: "coordinate tracking" },
];

const SECONDARY_PRESETS: Preset[] = [
  { id: "contour_trace", name: "Ghost Trace", description: "edge silhouette" },
  { id: "data_body", name: "Matrix Mode", description: "data rain" },
  { id: "particle_silhouette", name: "Particle Cloud", description: "ethereal" },
  { id: "signal_map", name: "Signal Map", description: "" },
  { id: "numeric_aura", name: "Numeric Aura", description: "" },
  { id: "thermal_scan", name: "Thermal Scan", description: "" },
  { id: "face_scanner", name: "Face Scan", description: "" },
  { id: "grid_trace", name: "Grid Trace", description: "" },
  { id: "face_mesh", name: "Face Mesh", description: "" },
];

// Combined list for lookups
const ALL_PRESETS = [...MAIN_PRESETS, ...SECONDARY_PRESETS];

interface PresetPickerProps {
  value: string;
  onChange: (preset: string) => void;
  disabled?: boolean;
  
  // Sequence mode props
  mode: 'single' | 'sequence';
  onModeChange: (mode: 'single' | 'sequence') => void;
  sequence: string[];
  onSequenceChange: (seq: string[]) => void;
}

export function PresetPicker({ 
  value, 
  onChange, 
  disabled,
  mode,
  onModeChange,
  sequence,
  onSequenceChange
}: PresetPickerProps) {
  
  const handlePresetClick = (presetId: string) => {
    if (mode === 'single') {
      onChange(presetId);
    } else {
      // Sequence mode: append if under limit
      if (sequence.length < 4) {
        onSequenceChange([...sequence, presetId]);
      }
    }
  };

  const handleRemoveFromSequence = (index: number) => {
    const newSeq = [...sequence];
    newSeq.splice(index, 1);
    onSequenceChange(newSeq);
  };

  return (
    <div>
      <div className="flex justify-between items-center mb-4">
        <p className="text-text-muted text-xs font-mono uppercase tracking-widest">
          Choose Effect
        </p>
        
        {/* Mode Toggle */}
        <div className="bg-white/5 rounded-lg p-0.5 flex text-xs font-medium">
          <button
            onClick={() => onModeChange('single')}
            disabled={disabled}
            className={`px-3 py-1.5 rounded-md transition-all duration-200 ${
              mode === 'single' 
                ? 'bg-white/10 text-white shadow-sm' 
                : 'text-text-secondary hover:text-white'
            }`}
          >
            Single
          </button>
          <button
            onClick={() => onModeChange('sequence')}
            disabled={disabled}
            className={`px-3 py-1.5 rounded-md transition-all duration-200 ${
              mode === 'sequence' 
                ? 'bg-white/10 text-white shadow-sm' 
                : 'text-text-secondary hover:text-white'
            }`}
          >
            Sequence
          </button>
        </div>
      </div>

      {/* Sequence Timeline (only in sequence mode) */}
      {mode === 'sequence' && (
        <div className="animate-fade-in">
          <SequenceTimeline 
            sequence={sequence} 
            presets={ALL_PRESETS}
            onRemove={handleRemoveFromSequence}
          />
        </div>
      )}
      
      {/* Main 4 presets - 2x2 grid, compact */}
      <div className="grid grid-cols-2 gap-2 mb-2">
        {MAIN_PRESETS.map((preset) => {
          const isSelected = mode === 'single' ? value === preset.id : false;
          const isInSequence = mode === 'sequence' ? sequence.includes(preset.id) : false;
          
          return (
            <button
              key={preset.id}
              onClick={() => handlePresetClick(preset.id)}
              disabled={disabled || (mode === 'sequence' && sequence.length >= 4)}
              className={`
                py-2.5 px-3 rounded-lg text-left transition-all duration-200 group
                active:scale-[0.98] hover:scale-[1.01]
                disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100 disabled:active:scale-100
                ${isSelected
                  ? "bg-white/10 border border-white/30" 
                  : "bg-white/5 border border-transparent hover:bg-white/8"
                }
                ${isInSequence && mode === 'sequence' ? "ring-1 ring-accent/40" : ""}
              `}
            >
              <div className="flex justify-between items-start">
                <span className="block text-white text-sm font-medium">{preset.name}</span>
                {isInSequence && (
                  <span className="text-[10px] bg-accent/20 text-accent px-1.5 rounded-full font-mono">
                    {sequence.filter(id => id === preset.id).length}
                  </span>
                )}
              </div>
              <span className="block text-text-muted text-xs font-mono mt-0.5">
                {preset.description}
              </span>
            </button>
          );
        })}
      </div>
      
      {/* Secondary presets - flexible row */}
      <div className="flex flex-wrap gap-1.5 justify-center">
        {SECONDARY_PRESETS.map((preset) => {
          const isSelected = mode === 'single' ? value === preset.id : false;
          const isInSequence = mode === 'sequence' ? sequence.includes(preset.id) : false;

          return (
            <button
              key={preset.id}
              onClick={() => handlePresetClick(preset.id)}
              disabled={disabled || (mode === 'sequence' && sequence.length >= 4)}
              className={`
                py-2 px-3 rounded-lg text-center text-xs transition-all duration-200
                active:scale-[0.96] hover:scale-[1.03]
                disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100 disabled:active:scale-100
                flex-1 min-w-[70px] max-w-[100px]
                ${isSelected
                  ? "bg-white/10 border border-white/30 text-white" 
                  : "bg-white/5 border border-transparent text-text-secondary hover:bg-white/8 hover:text-white"
                }
                ${isInSequence && mode === 'sequence' ? "ring-1 ring-accent/40" : ""}
              `}
            >
              {preset.name}
            </button>
          );
        })}
      </div>
    </div>
  );
}
