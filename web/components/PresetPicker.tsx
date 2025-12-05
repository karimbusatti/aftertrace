"use client";

import { SequenceTimeline } from "./SequenceTimeline";
import { motion } from "framer-motion";

type Preset = {
  id: string;
  name: string;
  description: string;
  icon?: React.ReactNode;
};

const MAIN_PRESETS: Preset[] = [
  { 
    id: "codenet_overlay", 
    name: "CodeNet", 
    description: "feature network",
    icon: (
      <div className="relative w-6 h-6 flex items-center justify-center opacity-50">
        <div className="absolute w-1.5 h-1.5 bg-white rounded-full top-1 left-1" />
        <div className="absolute w-1.5 h-1.5 bg-white rounded-full bottom-1 right-1" />
        <div className="absolute w-1 h-1 bg-white/50 rounded-full top-1 right-2" />
        <svg className="absolute inset-0 w-full h-full text-white/30" viewBox="0 0 24 24">
          <path d="M6 6 L18 18 M6 6 L16 6" stroke="currentColor" strokeWidth="1" />
        </svg>
      </div>
    )
  },
  { 
    id: "motion_flow", 
    name: "Motion Flow", 
    description: "curving lines trace movement",
    icon: (
      <svg className="w-6 h-6 text-white/50" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
        <path d="M4 12C4 12 8 8 12 12C16 16 20 12 20 12" />
        <path d="M4 16C4 16 8 12 12 16C16 20 20 16 20 16" opacity="0.5" />
      </svg>
    )
  },
  { 
    id: "binary_bloom", 
    name: "Binary Bloom", 
    description: "0/1 silhouette",
    icon: (
      <div className="text-[8px] font-mono leading-none opacity-50 text-center">
        010<br/>101<br/>010
      </div>
    )
  },
  { id: "blob_track", name: "Blob Track", description: "coordinate tracking" },
];

const SECONDARY_PRESETS: Preset[] = [
  { id: "code_shadow", name: "CodeShadow", description: "ascii matrix" },
  { id: "contour_trace", name: "Ghost Trace", description: "edge silhouette" },
  { id: "data_body", name: "Matrix Mode", description: "data rain" },
  { id: "particle_silhouette", name: "Particle Cloud", description: "ethereal" },
  { id: "signal_map", name: "Signal Map", description: "" },
  { id: "thermal_scan", name: "Thermal Scan", description: "" },
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
  maxSlots: number;
  onMaxSlotsChange: (slots: number) => void;
  segmentDuration: number;
  onSegmentDurationChange: (duration: number) => void;
}

export function PresetPicker({ 
  value, 
  onChange, 
  disabled,
  mode,
  onModeChange,
  sequence,
  onSequenceChange,
  maxSlots,
  onMaxSlotsChange,
  segmentDuration,
  onSegmentDurationChange
}: PresetPickerProps) {
  
  const handlePresetClick = (presetId: string) => {
    if (mode === 'single') {
      onChange(presetId);
    } else {
      // Sequence mode: append if under limit
      if (sequence.length < maxSlots) {
        onSequenceChange([...sequence, presetId]);
      }
    }
  };
  
  // When maxSlots changes, trim sequence if needed
  const handleMaxSlotsChange = (newMax: number) => {
    onMaxSlotsChange(newMax);
    if (sequence.length > newMax) {
      onSequenceChange(sequence.slice(0, newMax));
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
          {/* Slot count & Duration selectors */}
          <div className="flex flex-col gap-3 mb-4">
            {/* Row 1: How many effects? */}
            <div className="flex items-center justify-between">
              <span className="text-text-muted text-xs">Effects in sequence:</span>
              <div className="flex gap-1">
                {[2, 3, 4, 5].map((num) => (
                  <button
                    key={num}
                    onClick={() => handleMaxSlotsChange(num)}
                    disabled={disabled}
                    className={`w-7 h-7 rounded-md text-xs font-medium transition-all duration-150
                      ${maxSlots === num 
                        ? 'bg-accent text-white' 
                        : 'bg-white/5 text-text-secondary hover:bg-white/10 hover:text-white'
                      }`}
                  >
                    {num}
                  </button>
                ))}
              </div>
            </div>

            {/* Row 2: Switch speed */}
            <div className="flex items-center justify-between">
              <span className="text-text-muted text-xs">Switch every:</span>
              <div className="flex gap-1">
                {[0.25, 0.5, 1, 2].map((dur) => (
                  <button
                    key={dur}
                    onClick={() => onSegmentDurationChange(dur)}
                    disabled={disabled}
                    className={`px-2 h-7 rounded-md text-xs font-medium transition-all duration-150 min-w-[3rem]
                      ${segmentDuration === dur 
                        ? 'bg-white/20 text-white border border-white/30' 
                        : 'bg-white/5 text-text-secondary hover:bg-white/10 hover:text-white border border-transparent'
                      }`}
                  >
                    {dur}s
                  </button>
                ))}
              </div>
            </div>
          </div>
          
          <SequenceTimeline 
            sequence={sequence} 
            presets={ALL_PRESETS}
            onRemove={handleRemoveFromSequence}
            maxSlots={maxSlots}
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
              disabled={disabled || (mode === 'sequence' && sequence.length >= maxSlots)}
              className={`
                relative overflow-hidden py-3 px-3 rounded-lg text-left transition-all duration-200 group
                active:scale-[0.98] hover:scale-[1.01]
                disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100 disabled:active:scale-100
                ${isSelected
                  ? "bg-white/10 border border-white/30" 
                  : "bg-white/5 border border-transparent hover:bg-white/8"
                }
                ${isInSequence && mode === 'sequence' ? "ring-1 ring-accent/40" : ""}
              `}
            >
              {/* Background Animation for Binary Bloom */}
              {preset.id === "binary_bloom" && (
                <div className="absolute inset-0 opacity-[0.03] pointer-events-none overflow-hidden">
                  <div className="animate-scroll-text text-[8px] leading-[8px] font-mono whitespace-pre">
                    {`01010101010101010101010101
10101010101010101010101010
01010101010101010101010101
10101010101010101010101010`.repeat(5)}
                  </div>
                </div>
              )}

              <div className="flex justify-between items-start relative z-10">
                <span className="block text-white text-sm font-medium">{preset.name}</span>
                
                {/* Sequence Badge */}
                {isInSequence && (
                  <span className="text-[10px] bg-accent/20 text-accent px-1.5 rounded-full font-mono">
                    {sequence.filter(id => id === preset.id).length}
                  </span>
                )}

                {/* Animated Icon */}
                {!isInSequence && preset.icon && (
                  <div className="text-white/40 group-hover:text-white/80 transition-colors">
                    {preset.id === "codenet_overlay" ? (
                      <motion.div 
                        animate={{ scale: [1, 1.1, 1] }}
                        transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
                      >
                        {preset.icon}
                      </motion.div>
                    ) : preset.id === "motion_flow" ? (
                      <motion.div 
                        animate={{ x: [0, 2, 0] }}
                        transition={{ duration: 3, repeat: Infinity, ease: "easeInOut" }}
                      >
                        {preset.icon}
                      </motion.div>
                    ) : (
                      preset.icon
                    )}
                  </div>
                )}
              </div>
              
              <span className="block text-text-muted text-xs font-mono mt-0.5 relative z-10">
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
              disabled={disabled || (mode === 'sequence' && sequence.length >= maxSlots)}
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
