"use client";

import { SequenceTimeline } from "./SequenceTimeline";

type Preset = {
  id: string;
  name: string;
  description: string;
};

// Effects grouped into ordered categories for a clean, organised picker.
const CATEGORIES: { label: string; presets: Preset[] }[] = [
  {
    label: "Point & Depth",
    presets: [
      { id: "point_cloud", name: "Point Cloud", description: "3D dotted scan" },
      { id: "crystallize", name: "Crystallize", description: "low-poly mosaic" },
      { id: "neon_glow", name: "Neon Glow", description: "neon outline" },
      { id: "light_trails", name: "Light Trails", description: "long-exposure glow" },
    ],
  },
  {
    label: "Tracking",
    presets: [
      { id: "blob_track", name: "Blob Track", description: "coordinate boxes" },
      { id: "codenet_overlay", name: "CodeNet", description: "feature network" },
      { id: "motion_trace", name: "Motion Flow", description: "flowing trails" },
      { id: "signal_map", name: "Signal Map", description: "data overlay" },
      { id: "ocular_overload", name: "Ocular Overload", description: "retinal glitch" },
    ],
  },
  {
    label: "Code & Data",
    presets: [
      { id: "binary_bloom", name: "Binary Bloom", description: "0/1 silhouette" },
      { id: "code_shadow", name: "Code Shadow", description: "ascii matrix" },
      { id: "data_body", name: "Matrix Mode", description: "data rain" },
      { id: "glyph_trace", name: "Glyph Trace", description: "ascii ink" },
      { id: "ascii_core", name: "ASCII Core", description: "white on black" },
    ],
  },
  {
    label: "Glitch & Signal",
    presets: [
      { id: "signal_bloom", name: "Signal Bloom", description: "lava distortion" },
      { id: "chromatic_ghost", name: "Chromatic Ghost", description: "rainbow trails" },
      { id: "slit_scan", name: "Slit Scan", description: "time warp" },
      { id: "tv_static", name: "TV Static", description: "subject to static" },
    ],
  },
  {
    label: "Artistic",
    presets: [
      { id: "halftone", name: "Halftone", description: "b&w dots" },
      { id: "ink", name: "Ink", description: "pen sketch" },
      { id: "dither_trace", name: "Dither Trace", description: "ink flow" },
      { id: "contour_trace", name: "Ghost Trace", description: "edge silhouette" },
      { id: "kaleidoscope", name: "Kaleidoscope", description: "mirror mandala" },
    ],
  },
];

// Combined list for lookups (timeline, etc.)
const ALL_PRESETS: Preset[] = CATEGORIES.flatMap((c) => c.presets);

interface PresetPickerProps {
  value: string;
  onChange: (preset: string) => void;
  disabled?: boolean;

  // Sequence / overlay mode props
  mode: 'single' | 'sequence' | 'overlay';
  onModeChange: (mode: 'single' | 'sequence' | 'overlay') => void;
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

  const isMulti = mode === 'sequence' || mode === 'overlay';

  const handlePresetClick = (presetId: string) => {
    if (mode === 'single') {
      onChange(presetId);
    } else {
      // Sequence / overlay: append if under the chosen limit.
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
            className={`px-3 py-1.5 rounded-md transition-all duration-200 ${mode === 'single'
              ? 'bg-white/10 text-white shadow-sm'
              : 'text-text-secondary hover:text-white'
              }`}
          >
            Single
          </button>
          <button
            onClick={() => onModeChange('sequence')}
            disabled={disabled}
            className={`px-3 py-1.5 rounded-md transition-all duration-200 ${mode === 'sequence'
              ? 'bg-white/10 text-white shadow-sm'
              : 'text-text-secondary hover:text-white'
              }`}
          >
            Sequence
          </button>
          <button
            onClick={() => onModeChange('overlay')}
            disabled={disabled}
            className={`px-3 py-1.5 rounded-md transition-all duration-200 ${mode === 'overlay'
              ? 'bg-white/10 text-white shadow-sm'
              : 'text-text-secondary hover:text-white'
              }`}
          >
            Overlay
          </button>
        </div>
      </div>

      {/* Multi-effect panel (sequence alternates over time; overlay stacks them) */}
      {isMulti && (
        <div className="animate-fade-in">
          <p className="text-text-muted text-[11px] mb-3 leading-relaxed">
            {mode === 'overlay'
              ? 'Tap effects to layer them on top of each other, all at once.'
              : 'Tap effects to chain them; the clip alternates between them over time.'}
          </p>

          {/* Slot count & Duration selectors */}
          <div className="flex flex-col gap-3 mb-4">
            {/* Row 1: How many effects? */}
            <div className="flex items-center justify-between">
              <span className="text-text-muted text-xs">
                {mode === 'overlay' ? 'Effects to overlay:' : 'Effects in sequence:'}
              </span>
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

            {/* Row 2: Switch speed (sequence only) */}
            {mode === 'sequence' && (
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
            )}
          </div>

          <SequenceTimeline
            sequence={sequence}
            presets={ALL_PRESETS}
            onRemove={handleRemoveFromSequence}
            maxSlots={maxSlots}
          />
        </div>
      )}

      {/* Effects, grouped into ordered categories for a clean, organised grid */}
      <div className="space-y-4">
        {CATEGORIES.map((cat) => (
          <div key={cat.label}>
            <p className="text-text-muted text-[10px] font-mono uppercase tracking-widest mb-2">
              {cat.label}
            </p>
            <div className="grid grid-cols-2 gap-2">
              {cat.presets.map((preset) => {
                const isSelected = mode === 'single' ? value === preset.id : false;
                const isInSequence = isMulti ? sequence.includes(preset.id) : false;
                const count = isInSequence ? sequence.filter((id) => id === preset.id).length : 0;

                return (
                  <button
                    key={preset.id}
                    onClick={() => handlePresetClick(preset.id)}
                    disabled={disabled || (isMulti && sequence.length >= maxSlots)}
                    className={`
                      relative overflow-hidden py-2.5 px-3 rounded-lg text-left transition-all duration-200
                      active:scale-[0.98] hover:scale-[1.01]
                      disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100 disabled:active:scale-100
                      ${isSelected
                        ? "bg-white/10 border border-white/30"
                        : "bg-white/5 border border-transparent hover:bg-white/[0.09]"
                      }
                      ${isInSequence ? "ring-1 ring-accent/50" : ""}
                    `}
                  >
                    <div className="flex justify-between items-center gap-1">
                      <span className="block text-white text-sm font-medium leading-tight truncate">
                        {preset.name}
                      </span>
                      {count > 0 && (
                        <span className="shrink-0 text-[10px] bg-accent/20 text-accent px-1.5 rounded-full font-mono">
                          {count}
                        </span>
                      )}
                    </div>
                    <span className="block text-text-muted text-[11px] font-mono mt-0.5 truncate">
                      {preset.description}
                    </span>
                  </button>
                );
              })}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
