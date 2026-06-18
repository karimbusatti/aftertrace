"use client";

import { SequenceTimeline } from "./SequenceTimeline";

type IconFamily = "dots" | "track" | "code" | "glitch" | "art" | "cursor";

type Preset = {
  id: string;
  name: string;
  description: string;
  fam: IconFamily;
};

// Effects grouped into ordered categories for a clean, organised picker.
const CATEGORIES: { label: string; presets: Preset[] }[] = [
  {
    label: "Point & Depth",
    presets: [
      { id: "point_cloud", name: "Point Cloud", description: "3D dotted scan", fam: "dots" },
      { id: "cursor_cloud", name: "Cursor Cloud", description: "pixel cursors", fam: "cursor" },
      { id: "crystallize", name: "Crystallize", description: "low-poly mosaic", fam: "art" },
      { id: "neon_glow", name: "Neon Glow", description: "neon outline", fam: "art" },
      { id: "light_trails", name: "Light Trails", description: "long-exposure", fam: "art" },
    ],
  },
  {
    label: "Tracking",
    presets: [
      { id: "blob_track", name: "Blob Track", description: "coordinate boxes", fam: "track" },
      { id: "codenet_overlay", name: "CodeNet", description: "feature network", fam: "track" },
      { id: "motion_trace", name: "Motion Flow", description: "flowing trails", fam: "track" },
      { id: "signal_map", name: "Signal Map", description: "data overlay", fam: "track" },
      { id: "ocular_overload", name: "Ocular", description: "retinal glitch", fam: "track" },
    ],
  },
  {
    label: "Code & Data",
    presets: [
      { id: "binary_bloom", name: "Binary Bloom", description: "0/1 silhouette", fam: "code" },
      { id: "code_shadow", name: "Code Shadow", description: "ascii matrix", fam: "code" },
      { id: "data_body", name: "Matrix Mode", description: "data rain", fam: "code" },
      { id: "glyph_trace", name: "Glyph Trace", description: "ascii ink", fam: "code" },
      { id: "ascii_core", name: "ASCII Core", description: "white on black", fam: "code" },
    ],
  },
  {
    label: "Glitch & Signal",
    presets: [
      { id: "signal_bloom", name: "Signal Bloom", description: "lava distortion", fam: "glitch" },
      { id: "chromatic_ghost", name: "Chromatic", description: "rainbow trails", fam: "glitch" },
      { id: "slit_scan", name: "Slit Scan", description: "time warp", fam: "glitch" },
      { id: "tv_static", name: "TV Static", description: "to static", fam: "glitch" },
    ],
  },
  {
    label: "Halftone & Ink",
    presets: [
      { id: "halftone", name: "Halftone", description: "black dots", fam: "dots" },
      { id: "blacktone", name: "Blacktone", description: "white dots", fam: "dots" },
      { id: "ink", name: "Ink", description: "pen sketch", fam: "art" },
      { id: "dither_trace", name: "Dither Trace", description: "ink flow", fam: "art" },
      { id: "contour_trace", name: "Ghost Trace", description: "edge silhouette", fam: "art" },
      { id: "kaleidoscope", name: "Kaleidoscope", description: "mirror mandala", fam: "art" },
    ],
  },
];

// Combined list for lookups (timeline, etc.)
const ALL_PRESETS: Preset[] = CATEGORIES.flatMap((c) => c.presets);

// Tiny monochrome icon per family - keeps every chip uniform.
function FamilyIcon({ fam }: { fam: IconFamily }) {
  const cls = "w-3.5 h-3.5 shrink-0";
  switch (fam) {
    case "dots":
      return (
        <svg className={cls} viewBox="0 0 16 16" fill="currentColor">
          <circle cx="3" cy="3" r="1.4" /><circle cx="8" cy="3" r="1.4" /><circle cx="13" cy="3" r="1.4" />
          <circle cx="3" cy="8" r="1.4" /><circle cx="8" cy="8" r="1.4" /><circle cx="13" cy="8" r="1.4" />
          <circle cx="3" cy="13" r="1.4" /><circle cx="8" cy="13" r="1.4" /><circle cx="13" cy="13" r="1.4" />
        </svg>
      );
    case "track":
      return (
        <svg className={cls} viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.3">
          <rect x="2" y="2" width="5" height="5" rx="1" /><rect x="9" y="9" width="5" height="5" rx="1" />
          <path d="M7 4.5 L9 9" strokeDasharray="1.5 1.5" />
        </svg>
      );
    case "code":
      return (
        <svg className={cls} viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.3">
          <path d="M6 4 L2 8 L6 12 M10 4 L14 8 L10 12" />
        </svg>
      );
    case "glitch":
      return (
        <svg className={cls} viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.3">
          <path d="M1 6 H6 V3 H10 V9 H15" /><path d="M1 11 H7" opacity="0.6" />
        </svg>
      );
    case "cursor":
      return (
        <svg className={cls} viewBox="0 0 16 16" fill="currentColor">
          <path d="M3 2 L3 13 L6 10 L8 14 L10 13 L8 9 L12 9 Z" />
        </svg>
      );
    default: // art
      return (
        <svg className={cls} viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.3">
          <path d="M8 1 L9.8 6.2 L15 8 L9.8 9.8 L8 15 L6.2 9.8 L1 8 L6.2 6.2 Z" />
        </svg>
      );
  }
}

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

      {/* Effects: ordered categories, uniform compact chips (icon + name) */}
      <div className="space-y-3">
        {CATEGORIES.map((cat) => (
          <div key={cat.label}>
            <p className="text-text-muted text-[9px] font-mono uppercase tracking-[0.2em] mb-1.5">
              {cat.label}
            </p>
            <div className="grid grid-cols-3 gap-1.5">
              {cat.presets.map((preset) => {
                const isSelected = mode === 'single' ? value === preset.id : false;
                const isInSequence = isMulti ? sequence.includes(preset.id) : false;
                const count = isInSequence ? sequence.filter((id) => id === preset.id).length : 0;

                return (
                  <button
                    key={preset.id}
                    onClick={() => handlePresetClick(preset.id)}
                    disabled={disabled || (isMulti && sequence.length >= maxSlots)}
                    title={preset.description}
                    className={`
                      relative h-12 px-1.5 rounded-lg flex flex-col items-center justify-center gap-0.5
                      transition-all duration-200 active:scale-[0.95] hover:scale-[1.03]
                      disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100 disabled:active:scale-100
                      ${isSelected
                        ? "bg-white/10 border border-white/30 text-white"
                        : "bg-white/5 border border-transparent text-text-secondary hover:bg-white/[0.09] hover:text-white"
                      }
                      ${isInSequence ? "ring-1 ring-accent/50" : ""}
                    `}
                  >
                    <span className={isSelected || isInSequence ? "text-accent" : "text-white/45"}>
                      <FamilyIcon fam={preset.fam} />
                    </span>
                    <span className="text-[10px] font-medium leading-none text-center truncate w-full">
                      {preset.name}
                    </span>
                    {count > 0 && (
                      <span className="absolute -top-1 -right-1 text-[9px] bg-accent text-white w-4 h-4 flex items-center justify-center rounded-full font-mono">
                        {count}
                      </span>
                    )}
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
